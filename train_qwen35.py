#!/usr/bin/env python3
"""LoRA fine-tuning of the Qwen3.5 LLM backbone on multimodal PDF-QA."""

import argparse
from pathlib import Path
import types
import sys, os

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from pdfqa_dataloader import (
    PDFQuestionDataset,
    prompt_to_messages,
    read_records,
)


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


import torch.nn.functional as F
from transformers import Trainer


class QwenTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        print("inputs['input_ids'].size()", inputs["input_ids"].size())

        outputs = model(
            **inputs,
            use_cache=False,
        )

        logits = outputs.logits

        # Qwen returned only the final K logits because of logits_to_keep.
        k = logits.shape[1]

        # Logit at sequence position t predicts token t+1.
        # Therefore for the final K logits, targets are the final K-1
        # real tokens plus one ignored final position.
        target_labels = torch.full(
            (labels.shape[0], k),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )

        if k > 1:
            target_labels[:, :-1] = labels[:, -(k - 1):]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_labels.reshape(-1),
            ignore_index=-100,
        )

        return (loss, outputs) if return_outputs else loss


class PDFQACollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, samples):
        sample = samples[0]

        if sample["error"]:
            raise RuntimeError(sample["error"])

        record = sample["record"]

        prompt_messages = prompt_to_messages(
            sample["prompt"],
            sample["images"],
        )

        full_messages = prompt_messages + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": str(record["answer"]),
                    }
                ],
            }
        ]

        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        inputs = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        inputs.pop("token_type_ids", None)

        labels = inputs["input_ids"].clone()

        prompt_length = prompt_inputs["input_ids"].shape[-1]
        full_length = inputs["input_ids"].shape[-1]

        labels[:, :prompt_length] = -100

        inputs["labels"] = labels

        # Number of positions needed for answer loss.
        answer_length = full_length - prompt_length

        print(
            f"id={record['qid']} "
            f"seq_len={full_length} "
            f"answer_len={answer_length} "
            f"images={len(sample['images'])}"
        )

        # +1 because causal LM loss predicts token t from position t-1.
        inputs["logits_to_keep"] = answer_length + 1

        return inputs


def find_llm_lora_targets(model):
    projection_names = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    targets = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        leaf_name = name.rsplit(".", 1)[-1]

        if leaf_name not in projection_names:
            continue

        # Only the text/LLM backbone.
        if "language_model" not in name:
            continue

        # Extra safety: never target vision modules.
        if "visual" in name or "vision" in name:
            continue

        targets.append(name)

    if not targets:
        raise RuntimeError(
            "Could not find Qwen3.5 language_model LoRA targets. "
            "Inspect model.named_modules() for this Transformers version."
        )

    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen35_lora"),
    )
    parser.add_argument(
        "--page-mode",
        choices=["selected", "all"],
        default="selected",
    )
    parser.add_argument(
        "--content-mode",
        choices=["image", "hybrid"],
        default="hybrid",
    )
    parser.add_argument(
        "--inline-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)

    parser.add_argument(
        "--attn-implementation",
        choices=[
            "flash_attention_2",
            "sdpa",
            "eager",
        ],
        default="sdpa",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected.")

    # Dataset
    records = read_records(args.questions)

    if args.limit > 0:
        records = records[: args.limit]

    dataset = PDFQuestionDataset(
        records,
        page_mode=args.page_mode,
        content_mode=args.content_mode,
        inline_images=args.inline_images,
        questions_path=args.questions,
    )
    print(f"Training examples: {len(dataset)}")

    # Processor
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=16 * 16 * 64,   # 16,384 pixels
    )

    # Base multimodal Qwen3.5
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    replace_patch_embed_with_linear(model)
    model.config.use_cache = False

    # Gradient checkpointing saves substantial activation memory.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )

    # Find ONLY LLM backbone modules.
    target_modules = find_llm_lora_targets(model)
    print(f"Adding LoRA to {len(target_modules)} language-model Linear modules.")

    # Show a few so we can verify the hierarchy.
    for name in target_modules[:10]:
        print("  ", name)
    if len(target_modules) > 10:
        print("   ...")

    # LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model.config.use_cache = False
    model = get_peft_model(
        model,
        lora_config,
    )
    model.config.use_cache = False

    # Safety check: absolutely no trainable visual parameters.
    visual_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and ("visual" in name or "vision" in name)
    ]

    if visual_trainable:
        raise RuntimeError(
            "Unexpected trainable visual parameters:\n"
            + "\n".join(visual_trainable[:20])
        )

    model.print_trainable_parameters()

    # Trainer
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
        optim="adamw_torch",
    )

    trainer = QwenTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PDFQACollator(processor),
    )

    trainer.train()

    # Save final LoRA adapter.
    final_dir = args.output_dir / "final"

    trainer.model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    print(f"LoRA adapter saved to {final_dir}")


if __name__ == "__main__":
    main()