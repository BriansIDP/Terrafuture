#!/usr/bin/env python3
"""Evaluate Qwen3.5 on PDF page-image questions. One question per generation."""

import argparse
import json
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
    ap.add_argument('--top-n', type=int, default=3)
    ap.add_argument('--candidates', type=int, default=100)
    ap.add_argument('--rerank-top', type=int, default=50)
    ap.add_argument('--reranker-model', default='Qwen/Qwen3-Reranker-0.6B')
    ap.add_argument("--database-root", type=Path, default=Path("data/Phase1PDFBase"))

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA GPU not detected; run on a GPU compute node.'
        )

    # Load question records only.
    # Each record specifies its own manifest path.
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

    # Store only user questions and model answers.
    # Never store retrieved PDF content in this history.
    conversation_history = []

    for rec in tqdm(records, desc="Multi-turn inference"):

        question = rec["question"]

        # -------------------------------------------------
        # 1. Retrieve evidence for the CURRENT turn only.
        # -------------------------------------------------
        if not args.disable_pdf:
            hits = retriever.search(
                query=question,
                candidates=args.candidates,
                rerank_top=args.rerank_top,
                top_n=args.top_n,
            )

            # Only this turn's record gets these retrieved pages.
            rec = dict(rec)
            rec["retrieved_pages"] = [
                {
                    "doc_id": hit["parent"]["doc_id"],
                    "pages": hit["parent"]["page_numbers"],
                }
                for hit in hits["results"]
            ]

        # -------------------------------------------------
        # 2. Build a dataset for the CURRENT turn only.
        # -------------------------------------------------
        dataset = PDFQuestionDataset(
            records=[rec],
            page_mode=args.page_mode,
            content_mode=args.content_mode,
            inline_images=args.inline_images,
            questions_path=args.questions,
            database_root=args.database_root,
            use_pdf=not args.disable_pdf,
        )

        sample = dataset[0]

        result = {
            "qid": rec["qid"],
            "question": question,
            "answer": rec.get("answer"),
            "manifest": rec.get("manifest"),
            "retrieved_pages": rec.get("retrieved_pages", None),
            "pred": None,
        }

        # -------------------------------------------------
        # 3. Construct the multi-turn model input.
        # -------------------------------------------------

        # Contains only the CURRENT question and its
        # retrieved PDF evidence, including page images.
        current_messages = prompt_to_messages(
            sample["prompt"],
            sample["images"],
        )

        # Previous turns contain plain question/answer text.
        # No previously retrieved PDF passages or images.
        messages = conversation_history + current_messages

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)
        import pdb; pdb.set_trace()

        # -------------------------------------------------
        # 4. Generate the response.
        # -------------------------------------------------
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=1.0,
                top_p=0.9,
            )

        new_ids = output_ids[
            0, inputs["input_ids"].shape[-1]:
        ]

        prediction = processor.decode(
            new_ids,
            skip_special_tokens=True,
        ).strip()

        result["pred"] = prediction

        print(f"\n[{rec['qid']}] Question: {question}")
        print(f"[{rec['qid']}] Answer: {prediction}")

        # -------------------------------------------------
        # 5. Update conversation history.
        # -------------------------------------------------

        # IMPORTANT:
        # Add the ORIGINAL question, not sample["prompt"].
        # sample["prompt"] contains retrieved PDF evidence.
        conversation_history.append({
            "role": "user",
            "content": question,
        })

        conversation_history.append({
            "role": "assistant",
            "content": prediction,
        })

        # -------------------------------------------------
        # 6. Cleanup and save.
        # -------------------------------------------------
        del inputs, output_ids, new_ids
        del sample, dataset, current_messages, messages

        torch.cuda.empty_cache()

        results.append(result)

        # Save after every turn to preserve progress.
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
