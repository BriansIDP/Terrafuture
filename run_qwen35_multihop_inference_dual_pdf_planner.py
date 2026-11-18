#!/usr/bin/env python3
"""Dual-source inference with a planner that inspects retrieved PDF page images.

All options from run_qwen35_multihop_inference_dual.py are supported. The
planner also receives the most recently retained page images, subject to a
separate page budget. The answer still receives the full retained evidence.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

import run_qwen35_multihop_inference_dual as dual
from pdfqa_dataloader import prompt_to_messages, read_manifest


def planner_messages(messages, evidence, roots, max_pages):
    """Add recent retained page images to the planner's text and excerpts."""
    selected = []
    for hit in reversed(evidence):
        for number in reversed(hit['pages']):
            if len(selected) >= max_pages:
                break
            selected.append((hit['source'], hit['parent']['doc_id'], int(number)))
        if len(selected) >= max_pages:
            break
    if not selected:
        return messages

    selected.reverse()
    cache, images = {}, []
    prompt = messages[0]['content'] + '\n\nRetrieved PDF page images:\n'
    for source, doc_id, number in selected:
        root = roots[source].resolve()
        manifest_path = (root / doc_id / 'manifest.json').resolve()
        if not manifest_path.is_relative_to(root):
            raise ValueError(f'Invalid document ID: {doc_id}')
        if manifest_path not in cache:
            manifest = read_manifest(manifest_path)
            cache[manifest_path] = {int(page['page']): page for page in manifest['pages']}
        if number not in cache[manifest_path]:
            raise ValueError(f'Page {number} not found in {source}/{doc_id}')
        with Image.open(cache[manifest_path][number]['path']) as page_image:
            images.append(page_image.convert('RGB').copy())
        prompt += f'\nSource {source}, document {doc_id}, page {number}:\n<image>\n'
    return prompt_to_messages(prompt, images)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--planner-max-pages', type=int, default=6,
                        help='Maximum recent PDF page images supplied to the planner')
    parser.add_argument('--paper-database-root', type=Path, default=Path('data/Phase1PDFBase'))
    parser.add_argument('--standards-database-root', type=Path, default=Path('data/Phase1Standards'))
    options, _ = parser.parse_known_args()
    if options.planner_max_pages < 0:
        parser.error('--planner-max-pages must be non-negative')
    roots = {'papers': options.paper_database_root, 'standards': options.standards_database_root}

    original_decide = dual.decide_next_hop
    original_generate = dual.generate_text

    def decide(model, processor, question, evidence, *args, **kwargs):
        def generate_with_pages(model, processor, messages, *gen_args, **gen_kwargs):
            messages = planner_messages(messages, evidence, roots, options.planner_max_pages)
            return original_generate(model, processor, messages, *gen_args, **gen_kwargs)

        dual.generate_text = generate_with_pages
        try:
            return original_decide(model, processor, question, evidence, *args, **kwargs)
        finally:
            dual.generate_text = original_generate

    # The shared CLI owns all other options. Show the extra option in its help.
    if '--help' in sys.argv or '-h' in sys.argv:
        print('Additional option: --planner-max-pages N  '
              'Maximum recent PDF page images supplied to the planner (default: 6).\n')
    # Remove both --planner-max-pages N and --planner-max-pages=N forms.
    args = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--planner-max-pages':
            i += 2
        elif arg.startswith('--planner-max-pages='):
            i += 1
        else:
            args.append(arg)
            i += 1
    saved_argv = sys.argv
    dual.decide_next_hop = decide
    try:
        sys.argv = [saved_argv[0], *args]
        dual.main()
    finally:
        dual.decide_next_hop = original_decide
        sys.argv = saved_argv


if __name__ == '__main__':
    main()
