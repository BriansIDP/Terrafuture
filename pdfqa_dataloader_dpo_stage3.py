#!/usr/bin/env python3
"""Reconstruct dual-source stage 3 rollout contexts for offline DPO.

The existing rollout format records page IDs, but not the retriever's matched
children. Planner evidence reconstructed from manifest text is therefore an
approximation of the evidence seen during generation.
"""
from pathlib import Path

from PIL import Image
from pdfqa_dataloader import make_pdf_prompt, prompt_to_messages, read_manifest
from torch.utils.data import Dataset

INSTRUCTIONS = (
    "Use the supplied research papers and structural design standards to answer "
    "the question. Distinguish research findings from normative requirements. "
    "Cite the document and page for substantive claims. Follow applicable "
    "conditions, exceptions and cross-references. If evidence is missing or "
    "unreadable, say what cannot be verified; do not invent requirements or values."
)


class PDFQuestionDataset(Dataset):
    def __init__(self, records, paper_database_root, standards_database_root,
                 page_mode='selected', content_mode='hybrid', inline_images=True,
                 questions_path=None, use_pdf=True):
        self.records = records
        self.roots = {'papers': Path(paper_database_root).resolve(),
                      'standards': Path(standards_database_root).resolve()}
        self.content_mode = content_mode
        self.inline_images = inline_images
        self.manifest_cache = {}
        if page_mode != 'selected' or not use_pdf:
            raise ValueError('Stage 3 rollouts require selected retrieved PDF pages')

    def __len__(self):
        return len(self.records)

    def pages_for(self, item):
        source, doc_id = item['source'], item['doc_id']
        if source not in self.roots:
            raise ValueError(f'Invalid source: {source}')
        root = self.roots[source]
        path = (root / doc_id / 'manifest.json').resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Invalid document ID: {doc_id}')
        if path not in self.manifest_cache:
            self.manifest_cache[path] = {int(p['page']): p for p in read_manifest(path)['pages']}
        result = []
        for number in item['pages']:
            page = dict(self.manifest_cache[path][int(number)])
            page['_doc_id'] = f'{source}/{doc_id}'
            result.append(page)
        return result

    def __getitem__(self, idx):
        record = self.records[idx]
        try:
            pages = [page for item in record['retrieved_pages'] for page in self.pages_for(item)]
            if not pages:
                return {'prompt': INSTRUCTIONS + '\n\n' + record['question'], 'images': [], 'error': None}
            prompt, items = make_pdf_prompt(record['question'], pages, self.content_mode, self.inline_images)
            images = []
            for item in items:
                with Image.open(item['path']) as im:
                    image = im.convert('RGB')
                    if item['type'] == 'crop':
                        image.thumbnail((224, 224), Image.Resampling.LANCZOS)
                    images.append(image.copy())
            return {'prompt': INSTRUCTIONS + '\n' + prompt, 'images': images, 'error': None}
        except Exception as exc:
            return {'error': f'{type(exc).__name__}: {exc}'}


def evidence_excerpts(dataset, evidence):
    excerpts = []
    for item in evidence:
        children = item.get('matched_children')
        if '_parent_excerpt' in item:
            excerpt = item['_parent_excerpt']
        elif children is not None:
            excerpt = '\n'.join(c['text'][:1200] for c in children[:2])
        else:
            texts = [block.get('text', '') for page in dataset.pages_for(item)
                     for block in page.get('content', []) if block.get('type') == 'text']
            excerpt = '\n'.join(texts)[:1800]
        excerpts.append(f"Source {item['source']}, document {item['doc_id']}, "
                        f"pages {item['pages']}:\n{excerpt or '[Page image only]'}")
    return '\n\n'.join(excerpts)


def reconstruct_planner_segments(dataset, record):
    """Rebuild the exact inference planner prompts from the logged retained hits."""
    question = record['question']
    trace = record.get('retrieval_trace')
    if not trace or trace[0].get('source') != 'standards':
        raise ValueError(f"{record['qid']}: expected an initial standards search")
    evidence, segments = [], []
    for i, hop in enumerate(trace):
        if 'new_pages' not in hop:
            raise ValueError(f"{record['qid']}: missing new_pages at hop {i + 1}")
        if i == 1 and hop.get('source') != 'papers':
            raise ValueError(f"{record['qid']}: second search must target papers")
        for item in hop['new_pages']:
            # Inference retains matched children only from the selected pages.
            # Older traces without raw_results fall back to manifest text.
            retained = dict(item)
            for hit in hop.get('raw_results', []):
                if hit.get('parent', {}).get('doc_id') == item['doc_id']:
                    pages = [int(p) for p in item['pages']]
                    children = [c for c in hit.get('matched_children', [])
                                if int(c['page']) in pages]
                    if children:
                        retained['matched_children'] = children
                    elif pages == hit['parent'].get('page_numbers'):
                        retained['_parent_excerpt'] = hit['parent'].get('text', '')[:1800]
                    else:
                        retained['matched_children'] = []
                    break
            evidence.append(retained)
        if 'planner' not in hop:
            continue
        planner = hop['planner']
        if not isinstance(planner, dict) or 'raw' not in planner:
            raise ValueError(f"{record['qid']}: missing raw planner output at hop {i + 1}")
        retrieval_option = (
            '{"decision":"RETRIEVE","source":"papers","query":"specific missing information"}. '
            'The only permitted retrieval source is papers.' if i == 0 else
            '{"decision":"RETRIEVE","source":"papers" or "standards",'
            '"query":"specific missing information"}. Choose a single source for each retrieval.'
        )
        prompt = (
            'Decide whether the evidence is sufficient to answer the engineering '
            'question. Consider missing research findings, applicable standard '
            'requirements, conditions, exceptions, annexes and tables. Do not answer '
            'the question. Return ONLY one JSON object: {"decision":"ANSWER"} or '
            + retrieval_option + '\n\nQuestion: ' + question + '\n\nEvidence:\n' +
            evidence_excerpts(dataset, evidence)
        )
        segments.append((prompt, planner['raw']))
    return segments
