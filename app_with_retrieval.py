#!/usr/bin/env python3
"""Single-GPU, multi-session PDF question-answering demo for Qwen3.5-9B."""
import json
import time
import types
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import fitz
import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor, TextIteratorStreamer
from peft import PeftModel

from retriever import PDFRetriever

from pdfqa_dataloader import make_pdf_prompt, prompt_to_messages, read_manifest

ROOT = Path(__file__).resolve().parent
MODEL_ID = os.environ.get('QWEN_MODEL', 'Qwen/Qwen3.5-9B')
LORA_PATH = os.environ.get('LORA_PATH', 'none')
MAX_PDF_MB = int(os.environ.get('MAX_PDF_MB', '50'))
MAX_PAGES = int(os.environ.get('MAX_PAGES', '200'))
MAX_IMAGE_EDGE = int(os.environ.get('MAX_IMAGE_EDGE', '1400'))
MAX_TEXT_PER_PAGE = int(os.environ.get('MAX_TEXT_PER_PAGE', '14000'))
# Optional prebuilt retrieval index (same index as the evaluation script).
INDEX_DIR = os.environ.get('PDF_INDEX_DIR', '')
DATABASE_ROOT = Path(os.environ.get('PDF_DATABASE_ROOT', 'data/Phase1PDFBase'))
RETRIEVER_MODEL = os.environ.get('PDF_RERANKER_MODEL', 'Qwen/Qwen3-Reranker-0.6B')
RETRIEVAL_TOP_N = int(os.environ.get('PDF_TOP_N', '5'))
RETRIEVAL_CANDIDATES = int(os.environ.get('PDF_CANDIDATES', '100'))
RETRIEVAL_RERANK_TOP = int(os.environ.get('PDF_RERANK_TOP', '50'))
WORK_ROOT = Path(os.environ.get('PDF_CHAT_WORKDIR', tempfile.gettempdir())) / 'qwen35_pdf_sessions'
WORK_ROOT.mkdir(parents=True, exist_ok=True)

_model = None
_processor = None
_gpu_lock = threading.Lock()
_retriever = None
_corpus_manifests = None


def replace_patch_embed_with_linear(model):
    """Always enable the provided GH200 Conv3d-to-Linear optimization.

    Only valid when the visual patch convolution is an independent, non-overlapping
    projection; fail at startup if the model architecture is incompatible.
    """
    patch_embed = model.model.visual.patch_embed
    conv = patch_embed.proj
    if not isinstance(conv, nn.Conv3d):
        raise TypeError(f'GH200 optimization expected Conv3d, got {type(conv).__name__}')
    if not (conv.kernel_size == conv.stride and conv.padding == (0, 0, 0)
            and conv.dilation == (1, 1, 1)):
        raise ValueError('GH200 optimization requires non-overlapping Conv3d patches')
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


def initialize_model():
    """Load and warm up the model BEFORE Gradio starts accepting requests."""
    global _model, _processor
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU found. Run on a GPU host.')
    started = time.perf_counter()
    print(f'[startup] Loading {MODEL_ID} onto the GPU...', flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=0,
        attn_implementation=os.environ.get('ATTN_IMPLEMENTATION', 'sdpa'),
    )
    print('[startup] Applying GH200 patch optimization (mandatory)...', flush=True)
    replace_patch_embed_with_linear(model)
    if os.path.exists(LORA_PATH):
        model = PeftModel.from_pretrained(model, LORA_PATH).eval()
    else:
        model = model.eval()

    # Use a real multimodal generation, rather than an unrelated random tensor,
    # to exercise the vision encoder, language model and token generation path.
    print('[startup] Warming up vision + text inference...', flush=True)
    warmup_image = Image.new('RGB', (256, 256), color=(110, 140, 170))
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': warmup_image},
        {'type': 'text', 'text': 'Describe this image briefly.'},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt', enable_thinking=False,
    ).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    print(f'[startup] Warmup finished in {time.perf_counter() - started:.1f}s; '
          'starting Gradio now.', flush=True)
    del inputs, output_ids, warmup_image
    _model, _processor = model, processor


def get_model():
    if _model is None or _processor is None:
        raise RuntimeError('Model was not initialized during startup.')
    return _model, _processor


def cleanup_session(documents):
    """Delete all document workspaces attached to the current chat."""
    for document in documents or []:
        if document.get('workspace'):
            shutil.rmtree(document['workspace'], ignore_errors=True)

def toggle_retrieval(enabled):
    enabled = not enabled
    label = (
        '📚 PDF Retrieval: ON'
        if enabled else '📚 PDF Retrieval: OFF'
    )
    return enabled, label


def stage_pdf(uploaded, pending_files):
    """Stage files in the composer; do NOT add a chatbot message until Send."""
    pending_files = list(pending_files or [])
    if uploaded:
        path = str(uploaded)
        if path not in pending_files:
            pending_files.append(path)
    return pending_files, staged_label(pending_files)


def staged_label(pending_files):
    if not pending_files:
        return 'No PDF attachments waiting to send.'
    names = ', '.join(f'`{Path(p).name}`' for p in pending_files)
    return f'📎 **Ready to send ({len(pending_files)}):** {names}'


def update_user_upload_bubble(history, turn_id, message):
    """Update ONLY this turn's user bubble; never create an assistant bubble."""
    history = list(history or [])
    marker = f'<!--pdf-turn:{turn_id}-->'
    for i in range(len(history) - 1, -1, -1):
        if history[i].get('role') == 'user' and marker in str(history[i].get('content', '')):
            history[i] = {'role': 'user', 'content': marker + message}
            return history
    raise ValueError('Pending user upload message was not found')


def process_one_pdf(path, documents, history, turn_id, base_text, completed, total):
    """Process one staged file; generator yields per-page progress."""
    workspace = None
    label = Path(path).name
    try:
        source = Path(path)
        if source.suffix.lower() != '.pdf':
            raise ValueError('Only PDF files are accepted.')
        if source.stat().st_size > MAX_PDF_MB * 1024 * 1024:
            raise ValueError(f'PDF exceeds {MAX_PDF_MB} MB limit.')
        with fitz.open(str(source)) as document:
            if document.needs_pass:
                raise ValueError('Password-protected PDFs are not supported.')
            pages = len(document)
            if not 1 <= pages <= MAX_PAGES:
                raise ValueError(f'This demo supports 1–{MAX_PAGES} pages (got {pages}).')
        workspace = Path(tempfile.mkdtemp(prefix='pdf_', dir=WORK_ROOT))
        destination = workspace / 'document.pdf'
        shutil.copyfile(source, destination)
        command = [sys.executable, '-u', str(ROOT / 'prepare_pdf.py'),
                   '--pdf', str(destination), '--out-dir', str(workspace / 'rendered'),
                   '--extract-content', '--dpi', '120', '--max-edge', str(MAX_IMAGE_EDGE)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                match = re.search(r'Page (\d+)/(\d+)', line)
                if match:
                    current, count = int(match.group(1)), int(match.group(2))
                    if current == 1 or current == count or current % max(1, count // 12) == 0:
                        status = f'⏳ **{label}**: processing page {current}/{count} ({completed + 1}/{total} PDFs)'
                        yield status, None
            stderr = process.stderr.read()
            if process.wait(timeout=30) != 0:
                raise RuntimeError(stderr[-1200:] or 'PDF preprocessing failed')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        manifest_path = workspace / 'rendered' / 'manifest.json'
        manifest = read_manifest(manifest_path)
        documents.append({'workspace': str(workspace), 'manifest_path': str(manifest_path),
                          'pages': len(manifest['pages']), 'name': label})
        yield f'✅ **{label}** ({pages} pages)', None
    except Exception as exc:
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)
        yield f'❌ **{label}**: {type(exc).__name__}: {exc}', str(exc)


def send_message(question, pending_files, documents, history):
    """Submit text + selected PDFs as ONE user turn, then preprocess on Send."""
    history = list(history or [])
    documents = list(documents or [])
    files = list(pending_files or [])
    query = (question or '').strip()
    if not query and not files:
        yield history, documents, question, pending_files, staged_label(files), '', 'Enter a message or attach a PDF.', ''
        return
    turn_id = uuid.uuid4().hex
    # Store the effective query for downstream Qwen inference.
    effective_query = query or 'Summarize the uploaded PDF(s).'
    header = query or '📄 PDF attachment(s)'
    file_lines = [f'📎 **{Path(path).name}** — ⏳ waiting to process' for path in files]
    def compose():
        return header + (('\n\n' + '\n'.join(file_lines)) if file_lines else '')
    history.append({'role': 'user', 'content': f'<!--pdf-turn:{turn_id}-->' + compose()})
    yield history, documents, '', [], staged_label([]), effective_query, 'Message sent.', ''
    errors = []
    for index, path in enumerate(files):
        file_lines[index] = f'📎 **{Path(path).name}** — ⏳ processing…'
        history = update_user_upload_bubble(history, turn_id, compose())
        yield history, documents, '', [], staged_label([]), effective_query, f'Processing {Path(path).name}…', ''
        for status, error in process_one_pdf(path, documents, history, turn_id, header, index, len(files)):
            file_lines[index] = '📎 ' + status
            history = update_user_upload_bubble(history, turn_id, compose())
            if error:
                errors.append(error)
            yield history, documents, '', [], staged_label([]), effective_query, status, ''
    summary = (f'{len(documents)} PDF(s) available: ' +
               ', '.join(f"{d['name']} ({d['pages']} pages)" for d in documents)) if documents else 'No PDFs attached.'
    if errors:
        summary += f' ({len(errors)} attachment(s) failed.)'
    yield history, documents, '', [], staged_label([]), effective_query, summary, ''


def clear_conversation(documents, history, model_history):
    """Stream clearing progress, then remove all PDFs and pending selections."""
    history = list(history or [])
    history.append({'role': 'assistant', 'content': '🧹 Clearing conversation and PDF files…'})
    yield documents, history, 'Clearing conversation and removing uploaded PDFs…', '', [], staged_label([]), '', '', model_history
    cleanup_session(documents)
    yield [], [], 'Chat cleared. No PDFs attached.', '', [], staged_label([]), '', '', []


STOPWORDS = set('a an and are as at be by can do does for from how in is it of on or the this to was were what when where which who why with using into about paper pdf'.split())


def retrieve_pages(manifest, query, count):
    """Small zero-dependency lexical retrieval over extracted PDF text.

    Supports multi-column PDFs imperfectly; the page images remain authoritative.
    """
    terms = [w for w in re.findall(r'[a-z0-9][a-z0-9_-]*', query.lower())
             if len(w) > 2 and w not in STOPWORDS]
    terms = list(dict.fromkeys(terms))
    pages = manifest['pages']
    if not terms:
        return pages[:count]
    scored = []
    for page in pages:
        content = ' '.join(b.get('text', '') for b in page.get('content', []) if b['type'] == 'text').lower()
        # Prefer pages matching several unique keywords over those repeating one term.
        score = sum(3 + min(content.count(t), 5) for t in terms if t in content)
        scored.append((score, page['page'], page))
    winners = sorted(scored, key=lambda item: (-item[0], item[1]))[:count]
    return sorted([p for _, _, p in winners], key=lambda p: p['page'])


def get_retriever():
    """Initialize an index-backed retriever lazily, under _gpu_lock."""
    global _retriever
    if not INDEX_DIR:
        return None
    if _retriever is None:
        _retriever = PDFRetriever(
            index_dir=Path(INDEX_DIR),
            reranker_model=RETRIEVER_MODEL,
            device='cuda',
        )
    return _retriever


def get_corpus_manifests():
    """Find page manifests under the indexed PDF database root once."""
    global _corpus_manifests
    if _corpus_manifests is None:
        _corpus_manifests = {}
        if DATABASE_ROOT.exists():
            for path in DATABASE_ROOT.rglob('manifest.json'):
                # Accept doc ids that correspond to a folder name, PDF stem,
                # or the folder containing the rendered output.
                aliases = {path.parent.name, path.parent.parent.name}
                for key in aliases:
                    _corpus_manifests.setdefault(key, path)
    return _corpus_manifests


def indexed_pages(question, documents):
    """Retrieve extra pages, excluding PDFs already uploaded in this session.

    Preserve retrieval-ranked document order, but present each document's pages
    in ascending PDF page order. RETRIEVAL_TOP_N limits only indexed evidence.
    """
    retriever = get_retriever()
    if retriever is None:
        return [], []
    hits = retriever.search(
        query=question,
        candidates=RETRIEVAL_CANDIDATES,
        rerank_top=RETRIEVAL_RERANK_TOP,
        top_n=RETRIEVAL_TOP_N,
    )
    manifest_map = get_corpus_manifests()
    uploaded_names = {
        alias.casefold()
        for doc in documents
        for alias in (doc['name'], Path(doc['name']).stem)
    }
    uploaded_manifests = {
        Path(doc['manifest_path']).resolve() for doc in documents
    }
    selected = []
    unresolved = []
    seen = set()
    for hit in hits['results']:
        parent = hit['parent']
        doc_id = str(parent['doc_id'])
        if (Path(doc_id).name.casefold() in uploaded_names or
                Path(doc_id).stem.casefold() in uploaded_names):
            continue  # Entire uploaded PDF supersedes its indexed counterpart.
        path = manifest_map.get(doc_id) or manifest_map.get(Path(doc_id).stem)
        if path is None:
            unresolved.append(doc_id)
            continue
        if Path(path).resolve() in uploaded_manifests:
            continue
        manifest = read_manifest(path)
        page_map = {int(p['page']): p for p in manifest['pages']}
        # Sort the pages of each matched document, regardless of hit ranking.
        for number in sorted({int(n) for n in parent['page_numbers']}):
            key = (str(Path(path).resolve()), number)
            if key in seen or number not in page_map:
                continue
            seen.add(key)
            selected.append((doc_id, page_map[number]))
            if len(selected) >= RETRIEVAL_TOP_N:
                break
        if len(selected) >= RETRIEVAL_TOP_N:
            break

    # Group by document in first-seen retrieval order; within each document,
    # present pages in their original numerical order.
    doc_order = {}
    for name, _ in selected:
        doc_order.setdefault(name, len(doc_order))
    selected.sort(key=lambda item: (doc_order[item[0]], int(item[1]['page'])))
    return selected, unresolved


def uploaded_pages(documents):
    """Return EVERY page of EVERY uploaded PDF, in upload and page order."""
    selected = []
    for doc in documents:
        manifest = read_manifest(doc['manifest_path'])
        for page in sorted(manifest['pages'], key=lambda p: int(p['page'])):
            selected.append((doc['name'], page))
    return selected


def make_current_evidence(question, selected):
    """Build a single-turn multimodal message. Do not cache it in history."""
    chunks = [
        'Answer using the supplied PDF evidence and conversation. '
        'Cite both filename and PDF page number. If the supplied PDF evidence '
        'does not support the answer, say so.\n',
        '\n### Start of supplied PDF documents\n',
    ]
    paths = []
    used = []
    for filename, page in selected:
        number = page['page']
        chunks.append(f'\nDocument: {filename}, page {number}:\n<image>\n')
        paths.append(page['path'])
        chunks.append(f'\nExtracted text from {filename}, page {number}:\n')
        for block in page.get('content', []):
            if block['type'] == 'text':
                chunks.append(block['text'][:MAX_TEXT_PER_PAGE] + '\n')
            elif block['type'] == 'image':
                chunks.append('[Figure crop]\n<image>\n')
                paths.append(block['path'])
        used.append(f'{filename} p.{number}')
    chunks.append('\n### End of supplied PDF documents\n')
    chunks.append('\nQuestion: ' + question + '\nAnswer:')
    images = []
    for image_path in paths:
        with Image.open(image_path) as im:
            img = im.convert('RGB')
            img.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
            images.append(img.copy())
    return prompt_to_messages(''.join(chunks), images), used


def ask(history, documents, max_new_tokens, question, model_history, retrieval_enabled):
    """Stream a turn using current retrieval + ALL previous plain Q/A turns."""
    history = list(history or [])
    documents = list(documents or [])
    model_history = list(model_history or [])
    if not (question or '').strip():
        yield history, 'Enter a message.', model_history
        return
    question = question.strip()
    answer_index = len(history)
    initial_status = (
        '🔎 Retrieving PDF evidence...'
        if retrieval_enabled
        else '🤖 Generating answer (retrieval disabled)...'
    )
    history.append({
        'role': 'assistant',
        'content': initial_status,
    })
    yield history, initial_status, model_history

    try:
        with _gpu_lock:
            model, processor = get_model()
            selected = []
            if retrieval_enabled:
                # Search on every turn; older evidence is never appended to messages.
                # Historical user questions help disambiguate follow-up queries.
                previous_questions = [m['content'] for m in model_history
                                    if m['role'] == 'user']
                retrieval_query = '\n'.join(previous_questions[-2:] + [question])
                # Uploaded PDFs take priority: include every page in its original
                # order. Append only non-uploaded indexed evidence afterward.
                uploaded = uploaded_pages(documents)
                retrieved, unresolved = indexed_pages(retrieval_query, documents)
                selected = uploaded + retrieved

                if selected:
                    current_messages, used = make_current_evidence(question, selected)
                    details = 'Current-turn PDF evidence: ' + ', '.join(used)
                else:
                    current_messages = [{'role': 'user', 'content': question}]
                    details = 'No PDF pages retrieved for this turn.'
                if unresolved:
                    details += (' | Unresolved indexed doc IDs: ' +
                                ', '.join(dict.fromkeys(unresolved) )[:400])
            else:
                # No retrieval, no PDF images, no extracted PDF text.
                current_messages = [
                    {'role': 'user', 'content': question}
                ]
                details = 'PDF retrieval disabled.'

            # Preserve full question + generated-answer history, not past PDF
            # evidence, images, or attachment/status text.
            messages = model_history + current_messages

            # Show retrieval status
            history[answer_index] = {
                'role': 'assistant',
                'content': (
                    '🤖 Generating answer using retrieved PDF evidence...'
                    if retrieval_enabled and selected
                    else '🤖 Generating answer...'
                ),
            }

            yield history, details, model_history

            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors='pt', enable_thinking=False,
            ).to(model.device)
            streamer = TextIteratorStreamer(
                processor.tokenizer, skip_prompt=True,
                skip_special_tokens=True, timeout=120)
            generation_errors = []

            def run_generation():
                try:
                    with torch.inference_mode():
                        model.generate(**inputs, max_new_tokens=int(max_new_tokens),
                                       do_sample=False, streamer=streamer)
                except BaseException as exc:
                    generation_errors.append(exc)
                    streamer.end()

            worker = threading.Thread(target=run_generation, daemon=True)
            worker.start()
            answer = ''
            try:
                for chunk in streamer:
                    answer += chunk
                    history[answer_index] = {'role': 'assistant', 'content': answer}
                    yield history, details, model_history
            finally:
                worker.join()
                del inputs
            if generation_errors:
                raise generation_errors[0]
            answer = answer.strip() or '(No text generated.)'
            history[answer_index] = {'role': 'assistant', 'content': answer}
            model_history = model_history + [
                {'role': 'user', 'content': question},
                {'role': 'assistant', 'content': answer},
            ]
            yield history, details, model_history
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        history[answer_index] = {'role': 'assistant', 'content':
                                 'GPU out of memory. Try a shorter question or smaller PDFs.'}
        yield history, 'Generation failed: GPU out of memory.', model_history
    except Exception as exc:
        history[answer_index] = {'role': 'assistant', 'content':
                                 f'Error: {type(exc).__name__}: {exc}'}
        yield history, 'Generation failed.', model_history


with gr.Blocks(title='Qwen3.5 PDF Chat') as demo:
    retrieval_enabled = gr.State(True)
    gr.Markdown('# Qwen3.5 Chat with PDFs\n'
                'Attach one or more PDFs, optionally enter a message, and click **Send**. '
                'PDFs appear as **user messages**, and remain available for later turns. '
                'PDF extraction always includes text and figure crops.')
    documents = gr.State([])
    pending_query = gr.State('')
    pending_files = gr.State([])
    model_history = gr.State([])  # Plain Q/A only; no retrieved PDF evidence.
    with gr.Row():
        clear = gr.Button('🧹 Clear chat and PDFs')
    status = gr.Markdown('No PDFs attached. You can start chatting now.')
    chat = gr.Chatbot(label='Conversation', type='messages', height=350,
                      group_consecutive_messages=False)
    with gr.Row():
        question = gr.Textbox(label='Message',
                              placeholder='Write a message and/or attach PDFs, then click Send…',
                              scale=8, lines=2)
        with gr.Row():
            attach = gr.UploadButton('📎 Attach PDF', file_types=['.pdf'],
                                    file_count='single', type='filepath', variant='secondary')
            submit = gr.Button('Send', variant='primary', scale=1)
        with gr.Row():
            retrieval_button = gr.Button('📚 PDF Retrieval: ON', variant='secondary')
    staged = gr.Markdown('No PDF attachments waiting to send.')
    tokens = gr.Slider(64, 8192, value=2048, step=64,
                       label='Maximum output tokens')
    details = gr.Markdown('')

    # Attaching stages the PDF in the composer only. The PDF is not processed,
    # and no chat bubble is created, until the user clicks Send.
    attach.upload(stage_pdf, [attach, pending_files], [pending_files, staged], queue=False)
    clear.click(clear_conversation, [documents, chat, model_history],
                [documents, chat, status, question, pending_files, staged,
                 pending_query, details, model_history], concurrency_limit=1, concurrency_id='pdf_chat')
    retrieval_button.click(
        toggle_retrieval,
        inputs=[retrieval_enabled],
        outputs=[retrieval_enabled, retrieval_button],
        queue=False,
    )
    # Both Enter and Send submit the same user turn (text + staged PDFs).
    for trigger in (submit.click, question.submit):
        event = trigger(send_message, [question, pending_files, documents, chat],
                        [chat, documents, question, pending_files, staged,
                         pending_query, status, details],
                        concurrency_limit=1, concurrency_id='pdf_chat')
        event.then(ask, [chat, documents, tokens, pending_query, model_history, retrieval_enabled],
                   [chat, details, model_history],
                   concurrency_limit=1, concurrency_id='pdf_chat')
    gr.Markdown('Each turn retrieves up to the configured page limit. Previous questions '
                'and answers remain in context; previous PDF evidence does not. '
                'Indexed corpus retrieval requires PDF_INDEX_DIR and PDF_DATABASE_ROOT.')

if __name__ == '__main__':
    initialize_model()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get('GRADIO_SERVER_NAME', '127.0.0.1'),
        server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7860')),
        show_error=True,
        max_file_size=f'{MAX_PDF_MB}mb',
        share=True,
    )