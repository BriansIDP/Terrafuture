#!/usr/bin/env python3
"""Independent Qwen3.5 questions with bounded multi-hop PDF retrieval."""

import argparse
import json
import re
import time
import types
from pathlib import Path
import sys, os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel

from pdfqa_dataloader import (
    read_records,
    read_manifest,
    PDFQuestionDataset,
    prompt_to_messages,
)
from retriever import PDFRetriever


def replace_patch_embed_with_linear(model):
    """Optional GH200 optimisation from supplied inference script."""
    patch_embed = model.model.visual.patch_embed
    conv = patch_embed.proj

    if not isinstance(conv, nn.Conv3d):
        raise TypeError(f'Expected Conv3d, got {type(conv).__name__}')

    if not (
        conv.kernel_size == conv.stride
        and conv.padding == (0, 0, 0)
        and conv.dilation == (1, 1, 1)
    ):
        raise ValueError(
            'Patch Conv3d is not equivalent to an independent patch Linear'
        )

    linear = nn.Linear(
        conv.weight[0].numel(),
        conv.out_channels,
        bias=conv.bias is not None,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )

    with torch.no_grad():
        linear.weight.copy_(conv.weight.flatten(1))
        if conv.bias is not None:
            linear.bias.copy_(conv.bias)

    patch_embed.proj = linear

    def fast_patch_forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.proj.in_features)
        return self.proj(hidden_states.to(self.proj.weight.dtype))

    patch_embed.forward = types.MethodType(
        fast_patch_forward, patch_embed
    )


def collate_fn(batch):
    """Keep samples as a list; each question is processed independently."""
    return batch[0]


def generate_text(model, processor, messages, max_new_tokens, do_sample=False):
    """Run one text or multimodal Qwen generation."""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=1.0,
            top_p=0.9,
        )
    text = processor.decode(
        output_ids[0, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()
    del inputs, output_ids
    return text


def decide_next_hop(model, processor, question, evidence, max_new_tokens):
    """Decide whether current text evidence is enough, or what to search next."""
    # Parent groups may contain three pages. Avoid feeding all of them repeatedly.
    excerpts = []
    for hit in evidence:
        parent = hit["parent"]
        matches = hit.get("matched_children", [])
        excerpt = "\n".join(c["text"][:1200] for c in matches[:2])
        if not excerpt:
            excerpt = parent["text"][:1800]
        excerpts.append(
            f"Document {parent['doc_id']}, pages {parent['page_numbers']}:\n{excerpt}"
        )
    planner_prompt = (
        "You are checking whether retrieved structural-standard excerpts contain ALL "
        "information needed to answer a question accurately. Account for applicable "
        "conditions, exceptions, referenced clauses, annexes and tables. "
        "Do not answer the engineering question. "
        "Return ONLY one JSON object: "
        '{"decision":"ANSWER"} or '
        '{"decision":"RETRIEVE","query":"specific missing clause, table or topic"}. '
        "Choose RETRIEVE if an important referenced requirement is missing. "
        "Choose ANSWER if the available excerpts are sufficient or if another "
        "retrieval is not needed.\n\n"
        f"Question: {question}\n\nEvidence:\n" + "\n\n".join(excerpts)
    )
    raw = generate_text(model, processor, [
        {"role": "user", "content": planner_prompt},
    ], max_new_tokens, do_sample=False)
    try:
        # Tolerate a model wrapping JSON in a Markdown fence or brief prose.
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        decision = json.loads(match.group(0) if match else raw)
        if decision.get("decision", "").upper() == "RETRIEVE":
            query = str(decision.get("query", "")).strip()
            if query:
                return "RETRIEVE", query
    except (ValueError, TypeError, AttributeError):
        pass
    return "ANSWER", ""


def main():
    ap = argparse.ArgumentParser()

    # No global manifest argument anymore.
    ap.add_argument(
        '--questions',
        type=Path,
        default=Path('questions.json'),
    )
    ap.add_argument(
        '--output',
        type=Path,
        default=Path('results/qwen35_pdfqa.json'),
    )
    ap.add_argument(
        '--model',
        default='Qwen/Qwen3.5-9B',
    )
    ap.add_argument(
        '--page-mode',
        choices=['selected', 'all'],
        default='selected',
    )
    ap.add_argument(
        '--content-mode',
        choices=['image', 'hybrid'],
        default='hybrid',
        help='image = pages only; hybrid = pages + extracted text/figures',
    )
    ap.add_argument(
        '--inline-images',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Insert extracted figure regions in hybrid prompt',
    )
    ap.add_argument(
        '--max-new-tokens',
        type=int,
        default=768,
    )
    ap.add_argument(
        '--max-pixels',
        type=int,
        default=128000,
    )
    ap.add_argument(
        '--attn-implementation',
        default='sdpa',
        choices=['sdpa', 'eager', 'flash_attention_2'],
    )
    ap.add_argument(
        '--num-workers',
        type=int,
        default=2,
    )
    ap.add_argument(
        '--prefetch-factor',
        type=int,
        default=1,
    )
    ap.add_argument(
        '--gh200-patch-optimization',
        action='store_true',
        help='Apply Conv3d-to-Linear patch embedding optimisation',
    )
    ap.add_argument(
        '--do-sample',
        action='store_true',
        help='Sample responses from model',
    )
    ap.add_argument(
        '--disable_pdf',
        action='store_true',
        help='Do not do retrieval',
    )
    ap.add_argument(
        '--ckpt',
        type=Path,
        default=None,
        help='Path to fine-tuned LoRA adapter checkpoint',
    )
    ap.add_argument('--index-dir', type=Path, required=True)
    ap.add_argument('--top-n', type=int, default=2)
    ap.add_argument('--max-hops', type=int, default=3,
                    help='Maximum retrieval rounds per question, including initial search')
    ap.add_argument('--max-evidence-pages', type=int, default=18,
                    help='Maximum PDF pages/images in final context')
    ap.add_argument('--decision-tokens', type=int, default=160)
    ap.add_argument('--candidates', type=int, default=100)
    ap.add_argument('--rerank-top', type=int, default=50)
    ap.add_argument('--reranker-model', default='Qwen/Qwen3-Reranker-0.6B')
    ap.add_argument("--database-root", type=Path, default=Path("data/Phase1Standards"))

    args = ap.parse_args()
    if args.max_hops < 1 or args.max_evidence_pages < 1:
        ap.error('--max-hops and --max-evidence-pages must be >= 1')

    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA GPU not detected; run on a GPU compute node.'
        )

    # Each question is independent; manifests are resolved by the dataloader.
    records = read_records(args.questions)

    ids = [str(r['qid']) for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate question IDs')

    processor = AutoProcessor.from_pretrained(
        args.model,
        max_image_size={'longest_edge': args.max_pixels},
    )

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation=args.attn_implementation,
    )
    if args.ckpt is not None and args.ckpt.exists():
        model = PeftModel.from_pretrained(
            model,
            str(args.ckpt),
            is_trainable=False,
        ).merge_and_unload()
    model.eval()

    if args.gh200_patch_optimization:
        replace_patch_embed_with_linear(model)

    # Initialize retriever once and reuse it across turns.
    retriever = None
    if not args.disable_pdf:
        retriever = PDFRetriever(
            index_dir=args.index_dir,
            reranker_model=args.reranker_model,
            device='cuda',
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for rec in tqdm(records, desc="Single-turn multi-hop RAG"):
        question = rec["question"]
        # -------------------------------------------------
        # 1. Multi-hop retrieval for this independent question.
        #    Search -> inspect evidence -> optionally search again.
        # -------------------------------------------------
        retrieval_trace = []
        if not args.disable_pdf:
            rec = dict(rec)
            query = question
            collected = {}  # (doc_id, parent_id) -> retrieval hit
            seen_queries = set()
            pages_used = 0

            for hop in range(args.max_hops):
                if query.casefold() in seen_queries:
                    break
                seen_queries.add(query.casefold())
                hits = retriever.search(
                    query=query,
                    candidates=args.candidates,
                    rerank_top=args.rerank_top,
                    top_n=args.top_n,
                )

                new_hits = 0
                for hit in hits["results"]:
                    parent = hit["parent"]
                    key = (parent["doc_id"], parent["parent_id"])
                    if key in collected:
                        continue
                    page_count = len(parent["page_numbers"])
                    if pages_used + page_count > args.max_evidence_pages:
                        continue
                    collected[key] = hit
                    pages_used += page_count
                    new_hits += 1

                print("Hop:", hop)
                print("Pages used:", pages_used)
                trace = {"hop": hop + 1, "query": query, "new_parents": new_hits,
                         "total_pages": pages_used}
                retrieval_trace.append(trace)
                if not collected or not new_hits or pages_used >= args.max_evidence_pages:
                    break
                if hop == args.max_hops - 1:
                    break

                decision, next_query = decide_next_hop(
                    model, processor, question,
                    list(collected.values()), args.decision_tokens,
                )
                trace["decision"] = decision
                if decision == "ANSWER":
                    print("Answer now!")
                    break
                else:
                    print("Next Query:", next_query)
                query = next_query

            # Preserve parent retrieval order, exactly as the existing loader expects.
            rec["retrieved_pages"] = [
                {"doc_id": hit["parent"]["doc_id"],
                 "pages": hit["parent"]["page_numbers"]}
                for hit in collected.values()
            ]

        # -------------------------------------------------
        # 2. Build a dataset for this question only.
        # -------------------------------------------------
        dataset = PDFQuestionDataset(
            records=[rec],
            page_mode=args.page_mode,
            content_mode=args.content_mode,
            inline_images=args.inline_images,
            questions_path=args.questions,
            database_root=args.database_root,
            use_pdf=not args.disable_pdf,
            standard_only=True,
        )

        sample = dataset[0]
        if sample["error"]:
            raise RuntimeError(f"Question {rec['qid']}: {sample['error']}")

        result = {
            "qid": rec["qid"],
            "question": question,
            "answer": rec.get("answer"),
            "manifest": rec.get("manifest"),
            "retrieved_pages": rec.get("retrieved_pages", None),
            "retrieval_trace": retrieval_trace,
            "pred": None,
        }

        # -------------------------------------------------
        # 3. Construct independent input for this question.
        # -------------------------------------------------

        # Contains this question and its
        # retrieved PDF evidence, including page images.
        current_messages = prompt_to_messages(
            sample["prompt"],
            sample["images"],
        )

        messages = current_messages

        # The final answer sees all accumulated page images and extracted content.
        prediction = generate_text(
            model, processor, messages, args.max_new_tokens, args.do_sample,
        )

        result["pred"] = prediction

        print(f"\n[{rec['qid']}] Question: {question}")
        print(f"[{rec['qid']}] Answer: {prediction}")

        # -------------------------------------------------
        # 6. Cleanup and save.
        # -------------------------------------------------
        del sample, dataset, current_messages, messages

        torch.cuda.empty_cache()

        results.append(result)

        # Save after every independent question to preserve progress.
        with open(args.output, "w", encoding="utf-8") as fout:
            json.dump(
                results,
                fout,
                indent=4,
                ensure_ascii=False,
            )

    del retriever

    print(f"Predictions saved to {args.output}")


if __name__ == '__main__':
    main()