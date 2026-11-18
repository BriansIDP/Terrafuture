#!/usr/bin/env python3
"""Online GRPO for standards-first, dual-source Qwen retrieval trajectories.

For each question, search standards and papers, then sample planner decisions
and an answer from the current LoRA policy. A frozen Qwen judge scores each
answer against the reference, and GRPO updates the sampled tokens online. No
rollout file is required. The judge and retrievers can use separate devices.
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoProcessor, Qwen3_5ForConditionalGeneration

import run_qwen35_multihop_inference_dual as dual
from LLM_as_a_judge import template as JUDGE_TEMPLATE
from pdfqa_dataloader import read_records
from retriever import PDFRetriever
from train_qwen35_dpo_stage3 import find_llm_lora_targets, replace_patch_embed_with_linear


def completion_logprobs(model, segment):
    """Log probabilities of the actual generated tokens, including EOS."""
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in segment['inputs'].items()}
    ids = inputs['input_ids']
    n = segment['completion_length']
    logits = model(**inputs, logits_to_keep=n + 1, use_cache=False).logits[:, :-1].float()
    return F.log_softmax(logits, dim=-1).gather(-1, ids[:, -n:].unsqueeze(-1)).squeeze(-1).squeeze(0)


def sample_text(model, processor, messages, max_new_tokens, segments, on_input=None):
    """Sample from the untruncated policy and save exact multimodal inputs."""
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            return_dict=True, return_tensors='pt', enable_thinking=False)
    inputs.pop('token_type_ids', None)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    prefix_length = inputs['input_ids'].shape[1]
    if on_input:
        on_input(prefix_length)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                temperature=1.0, top_p=1.0, top_k=0, use_cache=True)
    completion = output[:, prefix_length:]
    if completion.shape[1] == 0:
        raise RuntimeError('Model generated no tokens')
    inputs['input_ids'] = output
    if 'attention_mask' in inputs:
        mask = inputs['attention_mask']
        inputs['attention_mask'] = torch.cat((mask, mask.new_ones((mask.shape[0], completion.shape[1]))), dim=1)
    if 'mm_token_type_ids' in inputs:
        types = inputs['mm_token_type_ids']
        inputs['mm_token_type_ids'] = torch.cat((types, types.new_zeros((types.shape[0], completion.shape[1]))), dim=1)
    segments.append({'inputs': {k: v.detach().cpu() for k, v in inputs.items()},
                     'completion_length': completion.shape[1]})
    return processor.decode(completion[0], skip_special_tokens=True).strip()


def sample_trajectory(model, processor, record, retrievers, roots, args, step=None, candidate=None):
    segments = []
    hop = 0

    def log(message):
        if args.verbosity:
            print(f'step={step} candidate={candidate} {message}', flush=True)

    def on_hop(entry):
        nonlocal hop
        hop = entry['hop']
        log(f'hop={hop} source={entry["source"]}')
    # decide_next_hop uses this module-level generation function. Restore it even
    # on failure, so inference behavior elsewhere is unaffected.
    original = dual.generate_text
    dual.generate_text = lambda m, p, messages, limit, do_sample=False: sample_text(
        m, p, messages, limit, segments,
        lambda size: log(f'hop={hop} planner_input_tokens={size}'))
    try:
        planner = lambda q, e, papers_only: dual.decide_next_hop(
            model, processor, q, e, args.decision_tokens, True, papers_only)
        evidence, trace, stop = dual.retrieve_evidence(
            record['question'], retrievers, planner, args.max_hops, args.max_evidence_pages,
            {'papers': args.paper_top_n, 'standards': args.standards_top_n},
            args.candidates, args.rerank_top, on_hop=on_hop)
        messages = dual.build_messages(record['question'], evidence, roots, args.content_mode, args.inline_images)
        prediction = sample_text(model, processor, messages, args.max_new_tokens, segments,
                                 lambda size: log(f'hops={hop} answer_input_tokens={size}'))
    finally:
        dual.generate_text = original
    return {'segments': segments, 'pred': prediction, 'stop_reason': stop,
            'retrieved_pages': [{'source': h['source'], 'doc_id': h['parent']['doc_id'],
                                 'pages': h['pages']} for h in evidence],
            'retrieval_trace': trace}


class Judge:
    def __init__(self, model_name, device, attn):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, attn_implementation=attn).to(device).eval()
        self.device = device
        self.digits = []
        for digit in range(10):
            ids = self.processor.tokenizer.encode(str(digit), add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f'Judge tokenizer does not encode {digit} as a single token')
            self.digits.append(ids[0])
        if len(set(self.digits)) != 10:
            raise ValueError('Judge digit token IDs must be distinct')

    @torch.no_grad()
    def score(self, question, answer, prediction):
        prompt = JUDGE_TEMPLATE.format(question=question, answer=answer, prediction=prediction)
        inputs = self.processor.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors='pt', enable_thinking=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1, self.digits].float()
        return (torch.softmax(logits, dim=-1) * torch.arange(10, device=logits.device)).sum().item()


def set_policy_mode(model, training):
    model.train(training)
    # Sampling and scoring must use the same deterministic network, while train
    # mode enables gradient checkpointing for the backward pass.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


def update_group(model, optimizer, trajectories, advantages, args):
    """Token-level clipped GRPO with a frozen adapter-disabled KL reference."""
    # Snapshot old and reference probabilities before any update on this group.
    set_policy_mode(model, False)
    with torch.no_grad():
        for traj in trajectories:
            for seg in traj['segments']:
                seg['old_logps'] = completion_logprobs(model, seg).cpu()
                if args.beta:
                    with model.disable_adapter():
                        seg['ref_logps'] = completion_logprobs(model, seg).cpu()
    losses = []
    for _ in range(args.updates_per_group):
        optimizer.zero_grad(set_to_none=True)
        set_policy_mode(model, True)
        for traj, advantage in zip(trajectories, advantages):
            token_count = sum(seg['completion_length'] for seg in traj['segments'])
            for seg in traj['segments']:
                logp = completion_logprobs(model, seg)
                old = seg['old_logps'].to(logp.device)
                ratio = torch.exp((logp - old).clamp(-20, 20))
                clipped = ratio.clamp(1 - args.clip_epsilon, 1 + args.clip_epsilon)
                objective = torch.minimum(ratio * advantage, clipped * advantage)
                if args.beta:
                    log_ref_ratio = (seg['ref_logps'].to(logp.device) - logp).clamp(-20, 20)
                    objective = objective - args.beta * (torch.exp(log_ref_ratio) - log_ref_ratio - 1)
                loss = -objective.sum() / (token_count * len(trajectories))
                loss.backward()
                losses.append(loss.detach().item())
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.max_grad_norm)
        optimizer.step()
    set_policy_mode(model, False)
    return sum(losses) / args.updates_per_group


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--questions', type=Path, required=True, help='JSON/JSONL with qid, question, and reference answer')
    p.add_argument('--model', default='Qwen/Qwen3.5-9B')
    p.add_argument('--judge-model', help='Frozen reward model (defaults to the policy backbone)')
    p.add_argument('--policy-device', default='cuda:0')
    p.add_argument('--judge-device', default='cpu', help='For example cuda:1; defaults to CPU to avoid sharing GPU memory')
    p.add_argument('--retriever-device', default='cuda:0')
    p.add_argument('--paper-index-dir', type=Path, default=Path('data/phase1_vector_db'))
    p.add_argument('--standards-index-dir', type=Path, default=Path('data/phase1_standards_db'))
    p.add_argument('--paper-database-root', type=Path, default=Path('data/Phase1PDFBase'))
    p.add_argument('--standards-database-root', type=Path, default=Path('data/Phase1Standards'))
    p.add_argument('--reranker-model', default='Qwen/Qwen3-Reranker-0.6B')
    p.add_argument('--output-dir', type=Path, default=Path('results/qwen35_online_grpo_stage3'))
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--updates-per-group', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--beta', type=float, default=0.04)
    p.add_argument('--clip-epsilon', type=float, default=0.2)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--max-hops', type=int, default=3)
    p.add_argument('--paper-top-n', type=int, default=3)
    p.add_argument('--standards-top-n', type=int, default=3)
    p.add_argument('--max-evidence-pages', type=int, default=18)
    p.add_argument('--candidates', type=int, default=100)
    p.add_argument('--rerank-top', type=int, default=50)
    p.add_argument('--decision-tokens', type=int, default=160)
    p.add_argument('--max-new-tokens', type=int, default=768)
    p.add_argument('--max-pixels', type=int, default=128000)
    p.add_argument('--content-mode', choices=['image', 'hybrid'], default='hybrid')
    p.add_argument('--inline-images', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--lora-r', type=int, default=16)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--attn-implementation', choices=['sdpa', 'flash_attention_2', 'eager'], default='sdpa')
    p.add_argument('--judge-attn-implementation', choices=['sdpa', 'flash_attention_2', 'eager'], default='eager')
    p.add_argument('--save-steps', type=int, default=100)
    p.add_argument('--verbosity', type=int, choices=[0, 1], default=0,
                   help='0: step summaries only; 1: also show candidate hops and input token counts')
    p.add_argument('--seed', type=int, default=50)
    args = p.parse_args()
    if args.group_size < 2 or args.max_hops < 2 or min(args.epochs, args.updates_per_group,
        args.paper_top_n, args.standards_top_n, args.max_evidence_pages, args.candidates,
        args.rerank_top, args.decision_tokens, args.max_new_tokens, args.save_steps) < 1:
        p.error('group-size and max-hops must be at least 2; other count arguments must be positive')
    if not 0 < args.clip_epsilon < 1 or args.beta < 0 or args.learning_rate <= 0 or args.max_grad_norm <= 0:
        p.error('invalid optimization arguments')
    records = read_records(args.questions)
    if not records or any(not r.get('question') or not r.get('answer') or 'qid' not in r for r in records):
        p.error('each question must have qid, question, and a nonempty reference answer')
    if len({str(r['qid']) for r in records}) != len(records):
        p.error('question IDs must be unique')
    roots = {'papers': args.paper_database_root, 'standards': args.standards_database_root}
    for source, index in [('papers', args.paper_index_dir), ('standards', args.standards_index_dir)]:
        dual.validate_index(index, roots[source])
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, max_image_size={'longest_edge': args.max_pixels})
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn_implementation).to(args.policy_device)
    replace_patch_embed_with_linear(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.0, bias='none', task_type='CAUSAL_LM', target_modules=find_llm_lora_targets(model)))
    set_policy_mode(model, False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=0.0)
    retrievers = {source: PDFRetriever(index_dir=index, reranker_model=args.reranker_model,
                  device=args.retriever_device) for source, index in
                  [('papers', args.paper_index_dir), ('standards', args.standards_index_dir)]}
    judge = Judge(args.judge_model or args.model, args.judge_device, args.judge_attn_implementation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        order = list(records)
        random.shuffle(order)
        for record in order:
            trajectories = [sample_trajectory(model, processor, record, retrievers, roots, args,
                                              step=step + 1, candidate=i + 1)
                            for i in range(args.group_size)]
            scores = [judge.score(record['question'], record['answer'], t['pred']) for t in trajectories]
            mean = sum(scores) / len(scores)
            std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
            advantages = [(s - mean) / (std + 1e-8) for s in scores]
            loss = update_group(model, optimizer, trajectories, advantages, args) if std > 1e-8 else 0.0
            step += 1
            print(json.dumps({'epoch': epoch + 1, 'step': step, 'qid': record['qid'],
                              'scores': scores, 'loss': loss}), flush=True)
            if step % args.save_steps == 0:
                checkpoint = args.output_dir / f'checkpoint-{step}'
                model.save_pretrained(checkpoint)
                processor.save_pretrained(checkpoint)
    out = args.output_dir / 'final'
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f'LoRA saved in {out}', flush=True)


if __name__ == '__main__':
    main()
