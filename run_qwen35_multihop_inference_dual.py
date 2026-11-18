#!/usr/bin/env python3
"""Qwen multi-hop inference over separate paper and standards indexes.

The model decides whether to search, prioritizing unresolved standards and
their cross-references before research papers. Each source has its own search
limit. Build each index with retriever.py build against its database root.
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


def evidence_excerpts(evidence, max_passages=2, passage_chars=1200, parent_chars=1800):
    """Format only retained evidence for retrieval planning."""
    excerpts = []
    for hit in evidence:
        parent = hit['parent']
        matches = hit.get('matched_children', [])
        excerpt = '\n'.join(c['text'][:passage_chars] for c in matches[:max_passages])
        # Parent text can include pages excluded by the page budget. Avoid using it
        # unless the entire parent was retained.
        if not excerpt and hit['pages'] == parent['page_numbers']:
            excerpt = parent.get('text', '')[:parent_chars]
        excerpts.append(
            f"Source {hit['source']}, document {parent['doc_id']}, "
            f"pages {hit['pages']}:\n{excerpt or '[Page image only]'}"
        )
    return '\n\n'.join(excerpts)


def decide_next_hop(model, processor, question, evidence, max_new_tokens, do_sample=False,
                    available_sources=('standards', 'papers'), max_passages=2,
                    passage_chars=1200, parent_chars=1800):
    """Choose ANSWER or a retrieval source and query."""
    prompt = (
        'Decide the next step for this question. If it is straightforward or not '
        'relevant to a concrete design and can be answered without retrieval, '
        'choose ANSWER even when there is no evidence. For a concrete design, '
        'standards provide UK design guidance and requirements; papers provide '
        'material properties, research findings and evidence on novel designs. '
        'First identify applicable standards and resolve relevant cross-chapter '
        'references, conditions, exceptions, annexes and tables before moving '
        'on to papers. Search standards again with a targeted query when a '
        'reference or requirement remains unclear. Once applicable standards '
        'are sufficiently clear, search papers when material properties or '
        'new materials, findings or designs require research evidence; such '
        'paper evidence is often needed. If standards are not applicable, '
        'search papers directly when research evidence is needed. Choose ANSWER '
        'when the available evidence suffices or further searches are unlikely '
        'to help; do not assume missing requirements or research findings. '
        'Evidence excerpts may omit image-only content, which is not proof that '
        'a requirement is resolved. Do not answer the question. Return ONLY '
        'one JSON object: {"decision":"ANSWER"} or '
        '{"decision":"RETRIEVE","source":"papers" or "standards",'
        '"query":"specific missing information"}. Choose one source per search. '
        'Sources with remaining searches: ' + ', '.join(available_sources) + '. '
        'Only choose a source with remaining searches. If a necessary source '
        'is unavailable, use the remaining sources if useful or choose ANSWER '
        'and acknowledge missing evidence in the answer.'
        '\n\nQuestion: ' + question + '\n\nEvidence:\n' +
        (evidence_excerpts(evidence, max_passages, passage_chars, parent_chars)
         or '[No evidence retrieved]')
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
        if action == 'RETRIEVE' and source in available_sources and query:
            return {'decision': 'RETRIEVE', 'source': source, 'query': query, 'raw': raw}
    except (ValueError, TypeError, AttributeError):
        pass
    # Preserve malformed output in the trace; retrieval will proceed to answer.
    return {'decision': 'INVALID', 'raw': raw}


def retrieve_evidence(question, retrievers, decide, max_hops_by_source, max_pages,
                      top_n_by_source, candidates, rerank_top, progress=None, on_hop=None):
    """Let the planner decide whether and where to search at every step."""
    collected, seen_pages, trace = [], set(), []
    counts = {source: 0 for source in max_hops_by_source}
    stop_reason = 'max_hops'
    for hop in range(1, sum(max_hops_by_source.values()) + 1):
        available_sources = tuple(source for source, limit in max_hops_by_source.items()
                                  if counts[source] < limit)
        if progress:
            progress('Checking whether to search and which source to use.')
        decision = decide(question, collected, available_sources)
        if decision['decision'] != 'RETRIEVE':
            trace.append({'hop': hop - 1, 'planner': decision})
            stop_reason = 'answer' if decision['decision'] == 'ANSWER' else 'invalid_decision'
            break
        source, query = decision['source'], decision['query']
        if source not in available_sources:
            trace.append({'hop': hop - 1, 'planner': decision})
            stop_reason = 'invalid_decision'
            break
        counts[source] += 1
        if progress:
            progress(f'Search {hop}: searching {source} ({counts[source]}/'
                     f'{max_hops_by_source[source]}) for: {query}')
        hits = retrievers[source].search(query=query, candidates=candidates,
                                         rerank_top=rerank_top, top_n=top_n_by_source[source])['results']
        added = []
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
                if key not in seen_pages and len(seen_pages) < max_pages:
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
        entry = {'hop': hop, 'source': source, 'query': query, 'planner': decision,
                 'raw_results': [{'doc_id': hit['parent']['doc_id'],
                                  'pages': hit['parent']['page_numbers']}
                                 for hit in hits],
                 'new_pages': added, 'total_pages': len(seen_pages)}
        trace.append(entry)
        if on_hop:
            on_hop(entry)
        if progress:
            progress(f'Search {hop}: found {len(hits)} results, added '
                     f'{sum(len(item["pages"]) for item in added)} pages.')
        if hop == sum(max_hops_by_source.values()):
            stop_reason = 'max_hops'
            break
        if len(seen_pages) >= max_pages:
            stop_reason = 'max_evidence_pages'
            break
    if progress:
        progress(f'Retrieval stopped: {stop_reason}.')
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
    ap.add_argument('--max-standards-hops', type=int, default=4, help='Maximum standards searches')
    ap.add_argument('--max-paper-hops', type=int, default=4, help='Maximum paper searches')
    ap.add_argument('--max-evidence-pages', type=int, default=18)
    ap.add_argument('--decision-tokens', type=int, default=160)
    ap.add_argument('--planner-max-passages', type=int, default=2,
                    help='Maximum matching passages per retrieved result in the planner prompt')
    ap.add_argument('--planner-passage-chars', type=int, default=1200,
                    help='Maximum characters per matching passage in the planner prompt')
    ap.add_argument('--planner-parent-chars', type=int, default=1800,
                    help='Maximum characters of fallback parent text in the planner prompt')
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--max-pixels', type=int, default=128000)
    ap.add_argument('--content-mode', choices=['image', 'hybrid'], default='hybrid')
    ap.add_argument('--inline-images', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--attn-implementation', choices=['sdpa', 'eager', 'flash_attention_2'], default='sdpa')
    ap.add_argument('--gh200-patch-optimization', action='store_true')
    ap.add_argument('--do-sample', action='store_true')
    ap.add_argument('--seed', type=int, default=50)
    args = ap.parse_args()
    if min(args.max_standards_hops, args.max_paper_hops) < 0 or args.max_evidence_pages < 1:
        ap.error('source hop limits must be non-negative and --max-evidence-pages must be positive')
    top_n_by_source = {
        'papers': args.paper_top_n if args.paper_top_n is not None else args.top_n,
        'standards': args.standards_top_n if args.standards_top_n is not None else args.top_n,
    }
    if min(args.top_n, *top_n_by_source.values(), args.candidates, args.rerank_top, args.decision_tokens, args.max_new_tokens,
           args.planner_max_passages, args.planner_passage_chars, args.planner_parent_chars) < 1:
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
        planner = lambda q, e, sources: decide_next_hop(
            model, processor, q, e, args.decision_tokens, args.do_sample, sources,
            args.planner_max_passages, args.planner_passage_chars, args.planner_parent_chars)
        evidence, trace, stop_reason = retrieve_evidence(
            question, retrievers, planner,
            {'standards': args.max_standards_hops, 'papers': args.max_paper_hops}, args.max_evidence_pages,
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
