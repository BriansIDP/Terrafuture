# Terrafuture: Physics-Informed AI for Concrete Design

Terra1 contains the PDF question-answering, retrieval, training, evaluation, and demo code for Terrafuture. It uses Qwen multimodal models to answer questions using rendered research papers and structural design standards. Multi-hop inference can search a single document collection or both papers and standards, gathering evidence before producing an answer.

## Requirements and setup

Run commands from the repository root. The supplied job scripts expect a CUDA GPU, SLURM, a Conda environment named `qwen35`, and a Miniforge installation under `~/miniforge3`. They also contain site-specific `/projects/u6zu/Terrafuture` paths; adjust these, the input files, and SLURM resource requests for your system before submitting jobs. Set `HF_HOME` to a suitable model cache if needed.

The Python code uses PyTorch, Transformers with Qwen3.5 support, PEFT, Pillow, PyMuPDF, NumPy, tqdm, and Gradio. Retrieval additionally requires `sentence-transformers>=5.4`, FAISS, and `rank-bm25`. There is currently no pinned environment or requirements file; install versions compatible with your CUDA and model setup. Model weights may need to be downloaded from Hugging Face. Data, prepared PDFs, indexes, checkpoints, and generated results are not part of the tracked source files and must be supplied or generated separately.

## Prepare documents and build an index

Render each PDF into its own directory under a database root. The manifest records page images and extracted text and image blocks. Hybrid retrieval requires extracted content.

```bash
python prepare_pdf.py --pdf /path/to/paper.pdf \
  --out-dir data/Phase1PDFBase/paper --extract-content

python retriever.py build \
  --database data/Phase1PDFBase \
  --index-dir data/phase1_vector_db \
  --embedding-model Qwen/Qwen3-Embedding-4B

python retriever.py search --index-dir data/phase1_vector_db \
  --query "Your question" --top-n 3
```

Repeat preparation and indexing with a separate root and index for standards. The retriever embeds child text passages, combines vector and keyword retrieval, and returns reranked parent contexts. Rebuild an index when its document manifests change. `extract_pdf.sh` shows batch preparation of standards, with paths that must be adapted.

## Run inference and score results

Question files are JSON arrays or JSONL records. Dual-source inference expects a unique `qid` and a `question` for each record; include `answer` when scoring against a reference answer. For example:

```json
[{"qid": "example-1", "question": "What is the required strength?", "answer": "Reference answer"}]
```

After preparing both collections and their indexes, run on a CUDA node:

```bash
python run_qwen35_multihop_inference_dual.py \
  --questions data/questions.json --output results/dual.json \
  --paper-index-dir data/phase1_vector_db \
  --standards-index-dir data/phase1_standards_db \
  --paper-database-root data/Phase1PDFBase \
  --standards-database-root data/Phase1Standards \
  --max-hops 3 --paper-top-n 3 --standards-top-n 3

python LLM_as_a_judge.py --resultpath results/dual.json
```

Dual-source inference starts with one search of each collection, then may perform additional searches up to `--max-hops`. Its output includes predictions, retrieved pages, and a retrieval trace. The judge defaults to `Qwen/Qwen3.5-27B`, prints the average score, and writes a separate `_scored.json` file; it requires reference answers and sufficient model resources. `score.py` summarizes scored results, but its average-pages calculation assumes eight rollouts per question.

For a single collection, use `run_qwen35_multihop_inference.py`; other `run_qwen35_pdfqa*.py` scripts provide direct or retrieval-based variants. Use `--help` on each script for its options. `build_base.sh` and `run.sh` are site-specific SLURM examples; inspect their active commands and paths before running `sbatch`.

## Training

`train_qwen35.py` performs supervised LoRA training. `train_qwen35_dpo.py` trains on preference pairs derived from scored single-source rollouts, and `train_qwen35_dpo_stage3.py` supports dual-source trajectory training. The corresponding `pdfqa_dataloader*.py` files load questions and reconstruct PDF inputs. `train.sh` and `train_stage3.sh` provide SLURM examples with local data and checkpoint paths; update them before submission. `configs/ds_zero2.json` contains a DeepSpeed configuration.

## Interactive demo

`app_with_retrieval.py` starts a Gradio PDF chat application. `launch_demo_qwen35.sh` provides a SLURM launch example and configures the model, LoRA adapter, retrieval index, document root, and work directory through environment variables. Review its paths and `GRADIO_SHARE` setting before submission. The application defaults to `Qwen/Qwen3.5-9B`; `QWEN_MODEL`, `LORA_PATH`, `PDF_INDEX_DIR`, and `PDF_DATABASE_ROOT` can be used to configure it.

For a chat-focused alternative to Gradio, install Chainlit in the `qwen35` environment (`pip install chainlit`) and run `sbatch launch_chainlit.sh` on a GPU node. The app listens on `127.0.0.1:8000` by default; use an SSH tunnel to access that port from your browser, or set `CHAINLIT_HOST` and `CHAINLIT_PORT` for your deployment. It uses the same model, PDF processing, and retrieval configuration as the Gradio app. Attach PDFs directly in chat; Use the settings button to control retrieval, maximum output tokens, and maximum search hops; use `/clear` to clear the session. Each turn searches papers and standards first, then uses the model to plan further searches up to the hop limit. Configure the standards paths with `STANDARDS_INDEX_DIR` and `STANDARDS_DATABASE_ROOT`; `PDF_MAX_HOPS` sets the default hop limit. The model loads on the first question.
