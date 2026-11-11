#!/usr/bin/env python3
"""Hybrid PDF retrieval: embed child passages, return larger parent contexts.

Requires: pip install -U 'sentence-transformers>=5.4' faiss-cpu rank-bm25 numpy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_EMBED = 'Qwen/Qwen3-Embedding-4B'
DEFAULT_RERANK = 'Qwen/Qwen3-Reranker-0.6B'
TASK = 'Retrieve passages from scientific PDF documents that answer a user question.'


def tokenize(text):
    return re.findall(r'\w+', text.lower(), flags=re.UNICODE)


def chunk_text(text, max_words=350, overlap=80):
    """Create overlapping child passages without crossing page boundaries."""
    words = text.split()
    if not 0 <= overlap < max_words:
        raise ValueError('overlap must be >= 0 and < max_words')
    if not words:
        return []
    chunks = []
    step = max_words - overlap
    for start in range(0, len(words), step):
        part = words[start:start + max_words]
        if len(part) < 25 and chunks:
            chunks[-1] += ' ' + ' '.join(part)
        else:
            chunks.append(' '.join(part))
        if start + max_words >= len(words):
            break
    return chunks


def load_pages(root):
    """Discover per-PDF manifests, resolving image paths if database moved."""
    root = Path(root).resolve()
    manifests = sorted(root.rglob('manifest.json'))
    if not manifests:
        raise FileNotFoundError(f'No manifest.json found under {root}')
    pages = []
    seen = set()
    for manifest_file in manifests:
        data = json.loads(manifest_file.read_text(encoding='utf-8'))
        source_pdf = data.get('pdf', '')
        doc_id = str(manifest_file.parent.relative_to(root))
        if doc_id == '.':
            doc_id = Path(source_pdf).stem or 'document'
        if doc_id in seen:
            raise ValueError(f'Duplicate doc identifier: {doc_id}')
        seen.add(doc_id)
        for page in data.get('pages', []):
            image = Path(page['path'])
            if not image.is_absolute():
                image = manifest_file.parent / image
            if not image.is_file():
                image = manifest_file.parent / Path(page['path']).name
            texts = [block['text'] for block in page.get('content', [])
                     if block.get('type') == 'text' and block.get('text')]
            pages.append({
                'doc_id': doc_id, 'pdf': source_pdf,
                'page': int(page['page']), 'image': str(image.resolve()),
                'text': '\n'.join(texts),
            })
    return pages


def fingerprint(database):
    h = hashlib.sha256()
    for p in sorted(Path(database).rglob('manifest.json')):
        h.update(str(p.relative_to(database)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def build_index(database, index_dir, embedding_model=DEFAULT_EMBED,
                max_words=350, overlap=80, batch_size=8, parent_pages=3):
    """Create one parent per consecutive group of pages; embed only children."""
    import faiss
    from sentence_transformers import SentenceTransformer

    if parent_pages < 1:
        raise ValueError('parent_pages must be >= 1')
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    pages = load_pages(database)
    per_doc = defaultdict(list)
    for page in pages:
        per_doc[page['doc_id']].append(page)

    parents, children = [], []
    for doc_id, document_pages in per_doc.items():
        document_pages.sort(key=lambda p: p['page'])
        for start in range(0, len(document_pages), parent_pages):
            group = document_pages[start:start + parent_pages]
            parent_id = len(parents)
            parents.append({
                'parent_id': parent_id, 'doc_id': doc_id, 'pdf': group[0]['pdf'],
                'page_numbers': [p['page'] for p in group],
                'images': [p['image'] for p in group],
                'text': '\n\n'.join(f"[Page {p['page']}]\n{p['text']}" for p in group),
            })
            for page in group:
                for i, passage in enumerate(chunk_text(page['text'], max_words, overlap)):
                    children.append({
                        'parent_id': parent_id,
                        'doc_id': doc_id, 'pdf': page['pdf'],
                        'page': page['page'], 'image': page['image'],
                        'chunk': i, 'text': passage,
                    })

    if not children:
        raise ValueError('No extractable text. Image-only PDFs need OCR or visual retrieval.')

    encoder = SentenceTransformer(embedding_model, trust_remote_code=True)
    vectors = encoder.encode_document(
        [c['text'] for c in children], batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True,
    )
    vectors = np.ascontiguousarray(vectors.astype('float32'))
    dense = faiss.IndexFlatIP(vectors.shape[1])
    dense.add(vectors)
    faiss.write_index(dense, str(index_dir / 'dense.faiss'))
    for name, content in [('chunks.json', children), ('parents.json', parents),
                          ('pages.json', pages)]:
        (index_dir / name).write_text(json.dumps(content, ensure_ascii=False), encoding='utf-8')
    config = {
        'embedding_model': embedding_model, 'max_words': max_words,
        'overlap': overlap, 'parent_pages': parent_pages,
        'documents': len(per_doc), 'pages': len(pages), 'children': len(children),
        'parents': len(parents), 'manifest_fingerprint': fingerprint(database),
    }
    (index_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f"Indexed {len(per_doc)} PDFs, {len(pages)} pages, "
          f"{len(children)} child embeddings, {len(parents)} parents at {index_dir}")


class PDFRetriever:
    def __init__(self, index_dir, reranker_model=DEFAULT_RERANK, device=None):
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.dir = Path(index_dir)
        config = json.loads((self.dir / 'config.json').read_text(encoding='utf-8'))
        self.children = json.loads((self.dir / 'chunks.json').read_text(encoding='utf-8'))
        self.parents = json.loads((self.dir / 'parents.json').read_text(encoding='utf-8'))
        self.dense = faiss.read_index(str(self.dir / 'dense.faiss'))
        if self.dense.ntotal != len(self.children):
            raise ValueError('FAISS index and child metadata are out of sync')
        self.bm25 = BM25Okapi([tokenize(c['text']) for c in self.children])
        self.encoder = SentenceTransformer(config['embedding_model'], device=device,
                                           trust_remote_code=True)
        self.reranker = None
        if reranker_model and reranker_model.lower() != 'none':
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_model, device=device,
                                         trust_remote_code=True)

    def search(self, query, candidates=100, rerank_top=50, top_n=10):
        """Rank children; return top N distinct parents across all PDFs.

        A parent is ranked by its highest-ranked child. Reranking, when enabled,
        compares the question with the child passage, not the larger parent.
        """
        if not query.strip():
            raise ValueError('query cannot be empty')
        if top_n < 1 or candidates < 1 or rerank_top < 1:
            raise ValueError('top_n, candidates, rerank_top must be >= 1')
        n = min(candidates, len(self.children))
        emb = self.encoder.encode_query(query, prompt=TASK,
                                        normalize_embeddings=True,
                                        convert_to_numpy=True)
        emb = np.ascontiguousarray(np.asarray(emb, dtype=np.float32).reshape(1, -1))
        _, dense_ids = self.dense.search(emb, n)
        lexical_scores = self.bm25.get_scores(tokenize(query))
        lexical_ids = np.argsort(-lexical_scores, kind='stable')[:n]
        lexical_ids = [int(i) for i in lexical_ids if lexical_scores[i] > 0]
        fused = defaultdict(float)
        for ranked in (dense_ids[0], lexical_ids):
            for rank, idx in enumerate(ranked):
                if idx >= 0:
                    fused[int(idx)] += 1.0 / (60 + rank + 1)
        ids = sorted(fused, key=lambda i: fused[i], reverse=True)[:rerank_top]
        if self.reranker and ids:
            scores = np.asarray(self.reranker.predict(
                [(query, self.children[i]['text']) for i in ids],
                batch_size=4, show_progress_bar=False,
            )).reshape(-1)
            ids = [i for i, _ in sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)]

        parent_hits = {}
        for rank, i in enumerate(ids, 1):
            child = dict(self.children[i])
            child['rank'] = rank
            child['fusion_score'] = fused[i]
            pid = child['parent_id']
            if pid not in parent_hits:
                parent_hits[pid] = {'parent': self.parents[pid], 'matched_children': []}
            parent_hits[pid]['matched_children'].append(child)

        hits = list(parent_hits.values())[:top_n]
        return {
            'results': hits, 'num_pdfs': len({h['parent']['doc_id'] for h in hits}),
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--database', default='data/Phase1PDFBase')
    b.add_argument('--index-dir', default='data/phase1_vector_db')
    b.add_argument('--embedding-model', default=DEFAULT_EMBED)
    b.add_argument('--max-words', type=int, default=350)
    b.add_argument('--overlap', type=int, default=80)
    b.add_argument('--parent-pages', type=int, default=3)
    b.add_argument('--batch-size', type=int, default=8)
    s = sub.add_parser('search')
    s.add_argument('--index-dir', default='data/phase1_vector_db')
    s.add_argument('--query', required=True)
    s.add_argument('--reranker-model', default=DEFAULT_RERANK)
    s.add_argument('--candidates', type=int, default=100)
    s.add_argument('--rerank-top', type=int, default=50)
    s.add_argument('--top-n', type=int, default=10)
    args = p.parse_args()
    if args.command == 'build':
        build_index(args.database, args.index_dir, args.embedding_model,
                    args.max_words, args.overlap, args.batch_size, args.parent_pages)
    else:
        retriever = PDFRetriever(args.index_dir, args.reranker_model)
        print(json.dumps(retriever.search(args.query, args.candidates,
                                          args.rerank_top, args.top_n),
                         indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
