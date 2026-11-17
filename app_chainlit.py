#!/usr/bin/env python3
"""Chainlit interface for the local Qwen PDF chat and hybrid retriever."""
import asyncio
import queue
import threading

import chainlit as cl
from chainlit.input_widget import Slider, Switch

import app_with_retrieval as backend


@cl.on_chat_start
async def start_chat():
    cl.user_session.set('documents', [])
    cl.user_session.set('model_history', [])
    cl.user_session.set('retrieval_enabled', True)
    cl.user_session.set('max_tokens', 2048)
    cl.user_session.set('max_hops', backend.DEFAULT_MAX_HOPS)
    await cl.ChatSettings([
        Switch(id='retrieval_enabled', label='PDF retrieval', initial=True),
        Slider(id='max_tokens', label='Maximum output tokens', initial=2048,
               min=64, max=8192, step=64),
        Slider(id='max_hops', label='Maximum search hops', initial=backend.DEFAULT_MAX_HOPS,
               min=2, max=10, step=1,
               description='Includes the initial paper and standards searches.'),
    ]).send()
    await cl.Message(content=(
        'Ask a question or attach PDFs to explore them alongside the indexed corpus. '
        'Use the settings button to adjust retrieval, output tokens, and search hops. '
        'Use `/clear` to start over.'
    )).send()


@cl.on_settings_update
async def update_settings(settings):
    cl.user_session.set('retrieval_enabled', settings['retrieval_enabled'])
    cl.user_session.set('max_tokens', int(settings['max_tokens']))
    cl.user_session.set('max_hops', int(settings['max_hops']))


@cl.on_chat_end
async def end_chat():
    documents = cl.user_session.get('documents') or []
    await asyncio.to_thread(backend.cleanup_session, documents)


def run_turn(events, documents, files, question, model_history, retrieval_enabled, max_tokens, max_hops):
    """Run blocking PDF preparation and inference in one worker thread."""
    try:
        for file in files:
            path, name = file
            before = len(documents)
            for status, error in backend.process_one_pdf(path, documents, [], '', '', 0, len(files)):
                events.put(('status', status))
                if error:
                    events.put(('upload_error', error))
            if len(documents) > before:
                documents[-1]['name'] = name
        events.put(('documents', documents))
        with backend._gpu_lock:
            if backend._model is None:
                events.put(('status', 'Loading and warming up the model…'))
                backend.initialize_model()
        for history, details, updated_history in backend.ask(
                [], documents, max_tokens, question, model_history, retrieval_enabled, max_hops):
            events.put(('answer', (history[-1]['content'], details, updated_history)))
    except Exception as exc:
        events.put(('error', f'{type(exc).__name__}: {exc}'))
    finally:
        events.put(('done', None))


@cl.on_message
async def on_message(message: cl.Message):
    text = (message.content or '').strip()
    documents = cl.user_session.get('documents') or []
    if text == '/clear':
        await asyncio.to_thread(backend.cleanup_session, documents)
        cl.user_session.set('documents', [])
        cl.user_session.set('model_history', [])
        await cl.Message(content='Conversation context and uploaded PDFs cleared.').send()
        return
    files = [(element.path, element.name) for element in (message.elements or [])
             if getattr(element, 'path', None)]
    if not text and not files:
        await cl.Message(content='Enter a message or attach a PDF.').send()
        return
    question = text or 'Summarize the uploaded PDF(s).'
    events = queue.Queue()
    worker = threading.Thread(target=run_turn, args=(
        events, documents, files, question,
        cl.user_session.get('model_history') or [],
        cl.user_session.get('retrieval_enabled'),
        cl.user_session.get('max_tokens'),
        cl.user_session.get('max_hops'),
    ), daemon=True)
    worker.start()
    response = cl.Message(content='')
    sent = False
    displayed = ''
    details = ''
    updated_history = None
    async with cl.Step(name='PDF preparation and retrieval', type='retrieval') as step:
        while True:
            kind, value = await asyncio.to_thread(events.get)
            if kind == 'done':
                break
            if kind in ('status', 'upload_error'):
                step.output = value
                await step.update()
            elif kind == 'documents':
                cl.user_session.set('documents', value)
            elif kind == 'error':
                if not sent:
                    await response.send()
                    sent = True
                response.content = f'Error: {value}'
                await response.update()
            elif kind == 'answer':
                content, details, updated_history = value
                if content.startswith(('🔎 Retrieving', '🤖 Generating')):
                    step.output = details
                    await step.update()
                    continue
                if not sent:
                    await response.send()
                    sent = True
                if content.startswith(displayed):
                    await response.stream_token(content[len(displayed):])
                else:
                    response.content = content
                    await response.update()
                displayed = content
        step.output = details or step.output or 'Ready.'
    if updated_history is not None:
        cl.user_session.set('model_history', updated_history)
    if sent:
        await response.update()
