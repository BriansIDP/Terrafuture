#!/usr/bin/env python3
"""Render PDF pages and optionally extract ordered text + inline figures.

Extracted text comes from the PDF's text layer (no OCR). Figures are rendered
from image-block bounding boxes so charts/captions can be interleaved with text.
"""
import argparse
import hashlib
import json
from pathlib import Path
import io

import fitz
import pymupdf
from PIL import Image


def save_clip(page, clip, path, scale, quality):
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip,
                          alpha=False, colorspace=fitz.csRGB)
    with Image.open(io.BytesIO(pix.tobytes('png'))) as im:
        rgb = im.convert('RGB')
        rgb.save(path, 'JPEG', quality=quality, optimize=True)
        return rgb.size


def reading_order(blocks, page_width):
    """Approximate reading order for common 1/2-column research PDFs.

    Treat blocks covering the centre gutter as full-width separators. For
    non-spanning blocks, group columns (left before right). Reading order is
    heuristic; refer to the whole-page image for complex layouts.
    """
    mid = page_width / 2
    gutter = page_width * 0.04
    spanning = lambda b: b['bbox'][0] < mid-gutter and b['bbox'][2] > mid+gutter
    # Work in vertical bands separated by full-width blocks (heading/table).
    wide = sorted((b for b in blocks if spanning(b)), key=lambda b: b['bbox'][1])
    remaining = [b for b in blocks if not spanning(b)]
    result = []
    lower = float('-inf')
    for separator in wide + [None]:
        upper = separator['bbox'][1] if separator is not None else float('inf')
        band = [b for b in remaining if lower <= b['bbox'][1] < upper]
        result.extend(sorted(band, key=lambda b: (0 if (b['bbox'][0]+b['bbox'][2])/2 < mid else 1,
                                                   b['bbox'][1], b['bbox'][0])))
        if separator is not None:
            result.append(separator)
            lower = upper
    return result


def get_page_content(page, out_dir, page_number, scale, quality, min_image_area):
    blocks = page.get_text('dict', sort=False)['blocks']
    cleaned = []
    for b in blocks:
        bbox = b['bbox']
        if b['type'] == 0:
            text = '\n'.join(''.join(span['text'] for span in line.get('spans', []))
                             for line in b.get('lines', []))
            if text.strip():
                cleaned.append({'type': 'text', 'bbox': bbox, 'text': text.strip()})
        elif b['type'] == 1:
            rect = fitz.Rect(bbox) & page.rect
            if rect.is_empty or rect.get_area() < min_image_area:
                continue
            cleaned.append({'type': 'image', 'bbox': bbox})
    ordered = reading_order(cleaned, page.rect.width)
    result = []
    image_index = 0
    for block in ordered:
        if block['type'] == 'text':
            result.append({'type': 'text', 'text': block['text']})
        else:
            image_index += 1
            path = out_dir / f'page_{page_number:03}_figure_{image_index:03}.jpg'
            save_clip(page, fitz.Rect(block['bbox']) & page.rect, path, scale, quality)
            result.append({'type': 'image', 'path': str(path.resolve()),
                           'bbox': [round(x, 2) for x in block['bbox']]})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=Path('data/rendered'))
    ap.add_argument('--dpi', type=int, default=130)
    ap.add_argument('--max-edge', type=int, default=1800)
    ap.add_argument('--jpeg-quality', type=int, default=90)
    ap.add_argument('--extract-content', action=argparse.BooleanOptionalAction, default=True,
                    help='Store reading-order text and embedded image regions in manifest')
    ap.add_argument('--min-image-area', type=float, default=400.0,
                    help='Minimum source-PDF image block area in square points')
    args = ap.parse_args()
    if not args.pdf.is_file():
        ap.error(f'No PDF: {args.pdf}')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_hash = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    doc = fitz.open(args.pdf)
    manifest = {'pdf': str(args.pdf.resolve()), 'sha256': pdf_hash,
                'dpi': args.dpi, 'max_edge': args.max_edge,
                'content_extracted': args.extract_content, 'pages': []}
    for i, page in enumerate(doc, 1):
        scale = args.dpi / 72.0
        if args.max_edge:
            scale = min(scale, args.max_edge / max(page.rect.width, page.rect.height))
        path = args.out_dir / f'page_{i:03}.jpg'
        w, h = save_clip(page, page.rect, path, scale, args.jpeg_quality)
        entry = {'page': i, 'path': str(path.resolve()), 'width': w, 'height': h}
        if args.extract_content:
            entry['content'] = get_page_content(page, args.out_dir, i, scale,
                                                args.jpeg_quality, args.min_image_area)
        manifest['pages'].append(entry)
        print(f'Page {i}/{len(doc)} -> {path} ({len(entry.get("content", []))} blocks)')
    location = args.out_dir / 'manifest.json'
    location.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Manifest: {location}')


if __name__ == '__main__':
    main()
