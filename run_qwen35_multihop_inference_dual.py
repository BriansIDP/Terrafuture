#!/usr/bin/env python3
"""Qwen multi-hop inference over separate paper and standards indexes.

Each question searches papers first and standards second. Both searches count
against --max-hops. Subsequent searches are chosen by the model. Build each
index with retriever.py build against its corresponding database root.
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor

from pdfqa_dataloader import make_pdf_prompt, prompt_to_messages, read_manifest, read_records
from retriever import PDFRetriever, fingerprint
from run_qwen35_multihop_inference import generate_text, replace_patch_embed_with_linear


INSTRUCTIONS = (
    "Use the supplied research papers and structural design standards to answer "
    "the question. Distinguish research findings from normative requirements. "
    "Cite the document and page for substantive claims. Follow applicable "
    "conditions, exceptions and cross-references. If evidence is missing or "
    "unreadable, say what cannot be verified; do not invent requirements or values."
)


def evidence_excerpts(evidence):
    """Format only retained evidence for retrieval planning."""
    excerpts = []
    for hit in evidence:
        parent = hit['parent']
        matches = hit.get('matched_children', [])
        excerpt = '\n'.join(c['text'][:1200] for c in matches[:2])
        # Parent text can include pages excluded by the page budget. Avoid using it
        # unless the entire parent was retained.
        if not excerpt and hit['pages'] == parent['page_numbers']:
            excerpt = parent.get('text', '')[:1800]
        excerpts.append(
            f"Source {hit['source']}, document {parent['doc_id']}, "
            f"pages {hit['pages']}:\n{excerpt or '[Page image only]'}"
        )
    return '\n\n'.join(excerpts)


def generate_standards_query(model, processor, question, evidence, max_new_tokens, do_sample=False):
    """Generate a standards search from the question and initial paper evidence."""
    prompt = (
        'Generate a search query for structural design standards needed to answer '
        'the engineering question. Use the paper evidence to identify relevant '
        'design provisions, terminology and referenced standards. Target the '
        'applicable requirements, clauses, tables, conditions or exceptions, '
        'rather than research findings. Do not answer the question or invent '
        'standard numbers or clauses. Return ONLY one JSON object: '
        '{"query":"specific standards requirements or topic"}.\n\nQuestion: '
        + question + '\n\nPaper evidence:\n' + evidence_excerpts(evidence)
    )
    raw = generate_text(model, processor, [{'role': 'user', 'content': prompt}],
                        max_new_tokens, do_sample=do_sample)
    try:
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        obj = json.loads(match.group(0) if match else raw)
        query = obj.get('query')
        if isinstance(query, str) and query.strip():
            return {'query': query.strip(), 'raw': raw}
    except (ValueError, TypeError, AttributeError):
        pass
    return {'query': question, 'raw': raw, 'fallback': True}


def decide_next_hop(model, processor, question, evidence, max_new_tokens, do_sample=False):
    """Choose ANSWER or a specific database and query using both sources."""
    prompt = (
        'Decide whether the evidence is sufficient to answer the engineering '
        'question. Consider missing research findings, applicable standard '
        'requirements, conditions, exceptions, annexes and tables. Do not answer '
        'the question. Return ONLY one JSON object: {"decision":"ANSWER"} or '
        '{"decision":"RETRIEVE","source":"papers" or "standards",'
        '"query":"specific missing information"}. Choose a single source for '
        'each retrieval.\n\nQuestion: ' + question + '\n\nEvidence:\n' +
        evidence_excerpts(evidence)
    )
    raw = generate_text(model, processor, [{'role': 'user', 'content': prompt}],
                        max_new_tokens, do_sample=do_sample)
    try:
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        obj = json.loads(match.group(0) if match else raw)
        action = str(obj.get('decision', '')).upper().strip()
        if action == 'ANSWER':
            return {'decision': 'ANSWER', 'raw': raw}
        source = str(obj.get('source', '')).lower().strip()
        query = str(obj.get('query', '')).strip()
        if action == 'RETRIEVE' and source in ('papers', 'standards') and query:
            return {'decision': 'RETRIEVE', 'source': source, 'query': query, 'raw': raw}
    except (ValueError, TypeError, AttributeError):
        pass
    # Do not silently treat malformed planner output as a valid ANSWER.
    return {'decision': 'INVALID', 'raw': raw}


def retrieve_evidence(question, retrievers, decide, standards_query, max_hops, max_pages,
                      top_n_by_source, candidates, rerank_top):
    """Run the two mandatory searches, then follow the planner within the budget."""
    collected, seen_pages, trace = [], set(), []
    source, query = 'papers', question
    stop_reason = 'max_hops'
    for hop in range(1, max_hops + 1):
        hits = retrievers[source].search(query=query, candidates=candidates,
                                         rerank_top=rerank_top, top_n=top_n_by_source[source])['results']
        added = []
        # Reserve at least one page for standards during the initial paper search.
        limit = max_pages - 1 if hop == 1 else max_pages
        for hit in hits:
            parent = hit['parent']
            pages = []
            # Prefer the pages containing the matching passages when a parent
            # must be truncated to fit the remaining budget.
            page_order = list(dict.fromkeys(
                [c['page'] for c in hit.get('matched_children', [])]
                + parent['page_numbers']))
            for number in page_order:
                key = (source, parent['doc_id'], int(number))
                if key not in seen_pages and len(seen_pages) < limit:
                    seen_pages.add(key)
                    pages.append(int(number))
            if not pages:
                continue
            selected = dict(hit)
            selected['source'] = source
            selected['pages'] = pages
            selected['matched_children'] = [c for c in hit.get('matched_children', [])
                                            if int(c['page']) in pages]
            collected.append(selected)
            added.append({'source': source, 'doc_id': parent['doc_id'], 'pages': pages})
        entry = {'hop': hop, 'source': source, 'query': query,
                 'new_pages': added, 'total_pages': len(seen_pages)}
        trace.append(entry)
        if hop == 1:
            generated = standards_query(question, collected)
            source, query = 'standards', generated['query']
            entry['decision'] = 'RETRIEVE'
            entry['next_source'] = source
            entry['next_query'] = query
            entry['standards_query_generation'] = generated
            continue
        if hop == max_hops:
            stop_reason = 'max_hops'
            break
        if len(seen_pages) >= max_pages:
            stop_reason = 'max_evidence_pages'
            break
        decision = decide(question, collected)
        entry['planner'] = decision
        entry['decision'] = decision['decision']
        if decision['decision'] == 'ANSWER':
            stop_reason = 'answer'
            break
        if decision['decision'] != 'RETRIEVE':
            # Malformed planner output is not an ANSWER. Continue with a
            # deterministic fallback until the search budget is exhausted.
            source = 'papers' if source == 'standards' else 'standards'
            query = question
            entry['fallback'] = {'source': source, 'query': query}
        else:
            source, query = decision['source'], decision['query']
    return collected, trace, stop_reason


def build_messages(question, evidence, roots, content_mode, inline_images):
    """Resolve each retrieved page against its own database root."""
    pages, cache = [], {}
    for hit in evidence:
        source, doc_id = hit['source'], hit['parent']['doc_id']
        root = roots[source].resolve()
        manifest_path = (root / doc_id / 'manifest.json').resolve()
        if not manifest_path.is_relative_to(root):
            raise ValueError(f'Invalid document ID: {doc_id}')
        if manifest_path not in cache:
            manifest = read_manifest(manifest_path)
            cache[manifest_path] = {int(p['page']): p for p in manifest['pages']}
        for number in hit['pages']:
            if number not in cache[manifest_path]:
                raise ValueError(f'Page {number} not found in {source}/{doc_id}')
            page = dict(cache[manifest_path][number])
            page['_doc_id'] = f'{source}/{doc_id}'
            pages.append(page)
    if not pages:
        return [{'role': 'user', 'content': INSTRUCTIONS + '\n\n' + question}]
    prompt, image_items = make_pdf_prompt(question, pages, content_mode, inline_images)
    # Replace the single-PDF instruction with one that addresses both sources.
    prompt = INSTRUCTIONS + '\n' + prompt
    images = []
    for item in image_items:
        with Image.open(item['path']) as image:
            image = image.convert('RGB')
            if item['type'] == 'crop':
                image.thumbnail((224, 224), Image.Resampling.LANCZOS)
            images.append(image.copy())
    return prompt_to_messages(prompt, images)


def validate_index(index_dir, root):
    config_path = index_dir / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Index not found: {config_path}; build it with retriever.py build')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if not root.is_dir():
        raise FileNotFoundError(f'Database root not found: {root}')
    expected = config.get('manifest_fingerprint')
    if expected and expected != fingerprint(root):
        raise ValueError(f'Index {index_dir} does not match manifests under {root}; rebuild the index')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--questions', type=Path, default=Path('data/phase1_stage3_case_a_questions.json'))
    ap.add_argument('--output', type=Path, default=Path('results/qwen35_multihop_dual.json'))
    ap.add_argument('--paper-index-dir', type=Path, default=Path('data/phase1_vector_db'))
    ap.add_argument('--standards-index-dir', type=Path, default=Path('data/phase1_standards_db'))
    ap.add_argument('--paper-database-root', type=Path, default=Path('data/Phase1PDFBase'))
    ap.add_argument('--standards-database-root', type=Path, default=Path('data/Phase1Standards'))
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--ckpt', type=Path)
    ap.add_argument('--reranker-model', default='Qwen/Qwen3-Reranker-0.6B')
    ap.add_argument('--top-n', type=int, default=2,
                    help='Default ranked parent passages per search for either source')
    ap.add_argument('--paper-top-n', type=int, help='Ranked paper parent passages per search (overrides --top-n)')
    ap.add_argument('--standards-top-n', type=int, help='Ranked standards parent passages per search (overrides --top-n)')
    ap.add_argument('--candidates', type=int, default=100)
    ap.add_argument('--rerank-top', type=int, default=50)
    ap.add_argument('--max-hops', type=int, default=4, help='Total searches, including the mandatory first two')
    ap.add_argument('--max-evidence-pages', type=int, default=18)
    ap.add_argument('--decision-tokens', type=int, default=160)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--max-pixels', type=int, default=128000)
    ap.add_argument('--content-mode', choices=['image', 'hybrid'], default='hybrid')
    ap.add_argument('--inline-images', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--attn-implementation', choices=['sdpa', 'eager', 'flash_attention_2'], default='sdpa')
    ap.add_argument('--gh200-patch-optimization', action='store_true')
    ap.add_argument('--do-sample', action='store_true')
    ap.add_argument('--seed', type=int, default=50)
    args = ap.parse_args()
    if args.max_hops < 2 or args.max_evidence_pages < 2:
        ap.error('--max-hops and --max-evidence-pages must be >= 2 for both initial searches')
    top_n_by_source = {
        'papers': args.paper_top_n if args.paper_top_n is not None else args.top_n,
        'standards': args.standards_top_n if args.standards_top_n is not None else args.top_n,
    }
    if min(args.top_n, *top_n_by_source.values(), args.candidates, args.rerank_top, args.decision_tokens, args.max_new_tokens) < 1:
        ap.error('retrieval and generation counts must be positive')
    roots = {'papers': args.paper_database_root, 'standards': args.standards_database_root}
    indexes = {'papers': args.paper_index_dir, 'standards': args.standards_index_dir}
    for source in roots:
        validate_index(indexes[source], roots[source])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not detected; run on a GPU compute node.')
    if args.ckpt is not None and not args.ckpt.exists():
        raise FileNotFoundError(f'Adapter checkpoint not found: {args.ckpt}')
    records = read_records(args.questions)
    ids = [str(r['qid']) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate question IDs')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, max_image_size={'longest_edge': args.max_pixels})
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map='auto', attn_implementation=args.attn_implementation)
    if args.ckpt is not None:
        model = PeftModel.from_pretrained(model, str(args.ckpt), is_trainable=False).merge_and_unload()
    model.eval()
    if args.gh200_patch_optimization:
        replace_patch_embed_with_linear(model)
    retrievers = {source: PDFRetriever(index_dir=indexes[source], reranker_model=args.reranker_model,
                                      device='cuda') for source in roots}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for record in tqdm(records, desc='Dual-source multi-hop RAG'):
        question = record['question']
        planner = lambda q, e: decide_next_hop(model, processor, q, e, args.decision_tokens, args.do_sample)
        standards_planner = lambda q, e: generate_standards_query(
            model, processor, q, e, args.decision_tokens, args.do_sample)
        evidence, trace, stop_reason = retrieve_evidence(
            question, retrievers, planner, standards_planner, args.max_hops, args.max_evidence_pages,
            top_n_by_source, args.candidates, args.rerank_top)
        messages = build_messages(question, evidence, roots, args.content_mode, args.inline_images)
        prediction = generate_text(model, processor, messages, args.max_new_tokens, args.do_sample)
        results.append({
            'qid': record['qid'], 'question': question, 'answer': record.get('answer'),
            'retrieved_pages': [{'source': h['source'], 'doc_id': h['parent']['doc_id'],
                                 'pages': h['pages']} for h in evidence],
            'retrieval_trace': trace, 'stop_reason': stop_reason, 'pred': prediction,
        })
        with args.output.open('w', encoding='utf-8') as fout:
            json.dump(results, fout, indent=4, ensure_ascii=False)
        del messages
        torch.cuda.empty_cache()
    print(f'Predictions saved to {args.output}')


if __name__ == '__main__':
    main()
