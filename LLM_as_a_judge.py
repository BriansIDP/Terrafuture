
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



template = """You are an expert in concrete Engineering. You will be given a question, a reference answer and a predicted answer. You should provide your judgments on how the predicted answer correctly answered the question.
Question:
{question}

Reference Answer:
{answer}

Predicted Answer:
{prediction}

Give your score as a value between 0 and 9, where 0 means completely incorrect and 9 means exactly correct. Only give the number. Do not answer anything else.
"""

score_value = [i for i in range(10)]

def replace_patch_embed_with_linear(model):
    patch_embed = model.model.visual.patch_embed
    conv = patch_embed.proj
    linear = nn.Linear(
        conv.weight[0].numel(), conv.out_channels,
        bias=conv.bias is not None,
        device=conv.weight.device, dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(conv.weight.flatten(1))
        if conv.bias is not None:
            linear.bias.copy_(conv.bias)
    patch_embed.proj = linear

    def fast_patch_forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.proj.in_features)
        return self.proj(hidden_states.to(self.proj.weight.dtype))

    patch_embed.forward = types.MethodType(fast_patch_forward, patch_embed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--resultpath', default="results/qwen35_pdfqa.json")
    p.add_argument('--model', default='Qwen/Qwen3.5-27B')
    args = p.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='eager',
    ).eval()
    replace_patch_embed_with_linear(model)
    with open(args.resultpath) as fin:
        data = json.load(fin)

    totalscore = 0
    for datapiece in tqdm(data):
        prompt = template.format(
            question=datapiece["question"],
            answer=datapiece["answer"],
            prediction=datapiece["pred"]
        )
        messages = [{"role": "user", "content": prompt}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
            enable_thinking=False,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        scores = torch.softmax(outputs.scores[0][0, 15:25], dim=-1).tolist()
        cumscore = 0
        for i, score in enumerate(scores):
            cumscore += score * i
        datapiece["score"] = cumscore
        totalscore += cumscore

    print(totalscore / len(data))
    with open(args.resultpath.replace(".json", "_scored.json"), "w") as fout:
        json.dump(data, fout, indent=4)