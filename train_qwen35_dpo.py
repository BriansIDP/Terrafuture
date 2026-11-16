#!/usr/bin/env python3
"""Offline trajectory-level DPO for Qwen3.5 multi-hop PDF-RAG.

Uses only the existing scored-rollout fields plus cumulative `retrieved_pages`
inside each `retrieval_trace` hop. The collator reconstructs planner prompts
from the PDFs/manifests at training time. No full planner prompts are logged.

IMPORTANT: The original planner consumes retriever `matched_children` excerpts.
These cannot be recovered exactly from PDF page IDs alone. We substitute a
bounded excerpt from the manifest's extracted page text. This is an approximate
observational state, NOT an exact replay of the original inference prompts.
For exact offline DPO, store compact `matched_children` text or replay a
version-pinned retriever with identical results/order.

Only actions actually generated are scored: initial query, post-hop decisions
explicitly logged in `retrieval_trace`, and final multimodal answer. Retrieval
results are conditions, NOT tokens in the log-probability sum.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import types
import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, Trainer, TrainingArguments

from pdfqa_dataloader_dpo import PDFQuestionDataset, prompt_to_messages, reconstruct_planner_segments


def replace_patch_embed_with_linear(model):
    """Optional GH200 optimisation from supplied inference script."""
    patch_embed = model.model.visual.patch_embed
    conv = patch_embed.proj
    if not isinstance(conv, nn.Conv3d):
        raise TypeError(f'Expected Conv3d, got {type(conv).__name__}')
    if not (conv.kernel_size == conv.stride and conv.padding == (0, 0, 0)
            and conv.dilation == (1, 1, 1)):
        raise ValueError('Patch Conv3d is not equivalent to an independent patch Linear')
    linear = nn.Linear(conv.weight[0].numel(), conv.out_channels,
                       bias=conv.bias is not None,
                       device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        linear.weight.copy_(conv.weight.flatten(1))
        if conv.bias is not None:
            linear.bias.copy_(conv.bias)
    patch_embed.proj = linear

    def fast_patch_forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.proj.in_features)
        return self.proj(hidden_states.to(self.proj.weight.dtype))

    patch_embed.forward = types.MethodType(fast_patch_forward, patch_embed)

def find_llm_lora_targets(model):
    leaves = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    result = []
    for name, module in model.named_modules():
        if (isinstance(module, nn.Linear)
            and name.rsplit(".", 1)[-1] in leaves
            and "language_model" in name
            and not any(x in name for x in ("visual", "vision"))):
            result.append(name)
    if not result:
        raise RuntimeError("No language_model LoRA targets found; check installed Transformers version")
    return result


def build_pairs(records, min_gap, max_pairs_per_qid=0):
    grouped = defaultdict(list)
    for r in records:
        required = ("qid", "question", "score", "retrieved_pages", "pred")
        missing = [k for k in required if k not in r]
        if missing:
            raise ValueError(f"Rollout {r.get('qid')} missing {missing}")
        grouped[r["qid"]].append(r)
    pairs = []
    for qid, group in grouped.items():
        qid_pairs = []
        # Generate all possible pairs for this QID
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["question"] != b["question"]:
                    raise ValueError(f"Question mismatch in qid {qid}")
                gap = abs(float(a["score"]) - float(b["score"]))
                if gap < min_gap:
                    continue
                win, lose = ((a, b) if float(a["score"]) > float(b["score"]) else (b, a))
                # Skip identical trajectories
                if (win.get("retrieval_trace", []) == lose.get("retrieval_trace", [])
                    and win["pred"] == lose["pred"]
                    and win["retrieved_pages"] == lose["retrieved_pages"]
                ):
                    continue
                qid_pairs.append((win, lose, gap))
        qid_pairs.sort(key=lambda x: x[2], reverse=True)
        if max_pairs_per_qid > 0:
            qid_pairs = qid_pairs[:max_pairs_per_qid]
        pairs.extend((win, lose) for win, lose, gap in qid_pairs)
        print(f"{qid}: selected {len(qid_pairs)} DPO pairs", flush=True)
    return pairs


class TrajectoryPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, ix):
        return self.pairs[ix]


class TrajectoryCollator:
    """Reconstruct planner evidence and final multimodal PDF context at training time."""
    def __init__(self, processor, questions_path, database_root, page_mode, content_mode, inline_images):
        self.processor = processor
        self.pdf_dataset = PDFQuestionDataset(
            records=[], page_mode=page_mode, content_mode=content_mode,
            inline_images=inline_images, questions_path=questions_path,
            database_root=database_root, use_pdf=True, standard_only=True,
        )

    def encode(self, prompt_messages, response):
        # Do not train on the prompt, PDF images, or retrieved passages.
        answer_msg = {"role": "assistant", "content": [{"type": "text", "text": response}]}
        common = dict(tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
        prefix = self.processor.apply_chat_template(
            prompt_messages, add_generation_prompt=True, **common)
        full = self.processor.apply_chat_template(
            prompt_messages + [answer_msg], add_generation_prompt=False, **common)
        prefix_ids = prefix["input_ids"][0]
        full_ids = full["input_ids"][0]
        length = prefix_ids.numel()
        if not torch.equal(prefix_ids, full_ids[:length]):
            raise ValueError("Chat template prefix mismatch; check Qwen thinking and message formatting")
        if full_ids.numel() <= length:
            raise ValueError("Empty assistant completion")
        full.pop("token_type_ids", None)
        return {"inputs": {k: v for k, v in full.items()}, "prefix_length": length}

    def encode_trajectory(self, record):
        segments = [
            self.encode([{"role": "user", "content": prompt}], response)
            for prompt, response in reconstruct_planner_segments(self.pdf_dataset, record)
        ]
        # For the final answer, use the actual final cumulative retrieved pages.
        # This follows the inference script's PDFQuestionDataset + prompt_to_messages.
        self.pdf_dataset.records = [record]
        one = self.pdf_dataset[0]
        if one["error"]:
            raise RuntimeError(f"{record['qid']}: {one['error']}")
        final_messages = prompt_to_messages(one["prompt"], one["images"])
        return {"planner": segments, "answer": self.encode(final_messages, record["pred"])}

    def __call__(self, examples):
        if len(examples) != 1:
            raise ValueError("Use per_device_train_batch_size=1")
        win, lose = examples[0]
        return {"chosen": self.encode_trajectory(win), "rejected": self.encode_trajectory(lose)}


def segment_logprob(model, segment):
    """Sum log probabilities of assistant tokens, not prompt/tool tokens."""
    device = next(model.parameters()).device
    data = {k: v.to(device) for k, v in segment["inputs"].items()}
    ids = data["input_ids"]
    n = ids.shape[1] - segment["prefix_length"]
    # Last n+1 logits contain predictions for the n completion tokens.
    logits = model(**data, logits_to_keep=n + 1, use_cache=False).logits
    next_token_logits = logits[:, :-1, :].float()
    target = ids[:, -n:]
    token_lp = F.log_softmax(next_token_logits, dim=-1).gather(
        -1, target.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum()


def trajectory_logprob(model, traj, answer_weight):
    planner_logp = sum((segment_logprob(model, s) for s in traj["planner"]))
    answer_logp = segment_logprob(model, traj["answer"])
    return planner_logp + answer_weight * answer_logp


class TrajectoryDPOTrainer(Trainer):
    def __init__(self, *args, beta=0.1, answer_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.answer_weight = answer_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pos, neg = inputs["chosen"], inputs["rejected"]
        # The adapter-disabled pretrained model is the frozen reference.
        # Compute reference first to avoid retaining unwanted gradients.
        with torch.no_grad(), model.disable_adapter():
            ref_pos = trajectory_logprob(model, pos, self.answer_weight)
            ref_neg = trajectory_logprob(model, neg, self.answer_weight)
        print('pos traj size:', pos["answer"]["inputs"]["input_ids"].size())
        print('neg traj size:', neg["answer"]["inputs"]["input_ids"].size())
        pol_pos = trajectory_logprob(model, pos, self.answer_weight)
        pol_neg = trajectory_logprob(model, neg, self.answer_weight)
        logit = (pol_pos - pol_neg) - (ref_pos - ref_neg)
        loss = -F.logsigmoid(self.beta * logit)
        return (loss, {"logit": logit.detach()}) if return_outputs else loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=Path, required=True, help="JSON list with 8 sampled/scored paths per question")
    p.add_argument("--questions", type=Path, required=True, help="Original question file; resolves PDF data")
    p.add_argument("--database-root", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--output-dir", type=Path, default=Path("results/qwen35_trajectory_dpo"))
    p.add_argument("--page-mode", choices=["selected", "all"], default="selected")
    p.add_argument("--content-mode", choices=["image", "hybrid"], default="hybrid")
    p.add_argument("--inline-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-score-gap", type=float, default=0.2)
    p.add_argument("--max-pairs-per-qid", type=int, default=10)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--answer-weight", type=float, default=1.0,
                   help="1.0 = standard trajectory log probability; <1 downweights long answers")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--attn-implementation", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    p.add_argument("--save-steps", type=int, default=200)

    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    records = json.loads(args.rollouts.read_text(encoding="utf-8"))
    pairs = build_pairs(records, args.min_score_gap, args.max_pairs_per_qid)
    if not pairs:
        raise ValueError("No preference pairs. Need >=2 scored, divergent rollouts per qid.")
    print(f"DPO training pairs: {len(pairs)}")

    processor = AutoProcessor.from_pretrained(args.model, max_image_size={"longest_edge": 128000})
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn_implementation)
    replace_patch_embed_with_linear(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # No Conv3d->Linear optimization here: retain original visual model as frozen.
    targets = find_llm_lora_targets(model)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=targets))
    if any(p.requires_grad and ("visual" in n or "vision" in n) for n, p in model.named_parameters()):
        raise RuntimeError("Vision backbone unexpectedly trainable")
    model.print_trainable_parameters()

    collator = TrajectoryCollator(processor, args.questions, args.database_root,
                                  args.page_mode, args.content_mode, args.inline_images)
    train_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=args.warmup, lr_scheduler_type="cosine", max_grad_norm=1.0,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        logging_steps=1, remove_unused_columns=False, dataloader_num_workers=2,
        report_to="none", optim="adamw_torch")
    trainer = TrajectoryDPOTrainer(
        model=model, args=train_args, train_dataset=TrajectoryPairDataset(pairs),
        data_collator=collator, beta=args.beta, answer_weight=args.answer_weight)
    trainer.train()
    out = args.output_dir / "final"
    trainer.model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"LoRA saved in {out}")


if __name__ == "__main__":
    main()