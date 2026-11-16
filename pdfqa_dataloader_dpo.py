#!/usr/bin/env python3
"""PDF-QA dataset supporting a different PDF/manifest for every question."""

import json
import time
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


STD_INSTRUCTIONS = """Use the supplied excerpts from structural design standards to answer the question.
Follow applicable conditions, exceptions, annexes and cross-references. Cite the document
and page for each substantive requirement. If evidence is missing or unreadable, say what
cannot be verified; never invent a clause, table value, equation or design criterion."""

INSTRUCTIONS = """You are provided with a pdf file. You can use the pdf to help you find the answer to the user query."""


def read_records(path):
    path = Path(path)

    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def read_manifest(path):
    """Load manifest and resolve image paths relative to the manifest."""

    path = Path(path).resolve()

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    for page in data["pages"]:
        for item in [page] + page.get("content", []):
            if not item.get("path"):
                continue

            p = Path(item["path"])

            if not p.exists():
                # First try filename next to the manifest.
                candidate = path.parent / p.name

                if candidate.exists():
                    p = candidate
                elif not p.is_absolute():
                    p = path.parent / p

            item["path"] = str(p.resolve())

    return data


def make_pdf_prompt(
    question,
    pages,
    content_mode="hybrid",
    inline_images=True,
    standard_only=False,
):
    if content_mode not in {"image", "hybrid"}:
        raise ValueError(
            "content_mode must be 'image' or 'hybrid'"
        )
    chunks = [
        STD_INSTRUCTIONS if standard_only else INSTRUCTIONS,
        "\n###Start of the PDF\n",
    ]
    image_items = []

    for page in pages:
        number = page["page"]
        if "_doc_id" in page:
            chunks.append(f"\nDocument: {page['_doc_id']}\n")
        # Whole-page image
        chunks.append(
            f"\nPage {number}:\n<image>\n"
        )
        image_items.append({
            'path': page['path'],
            'type': 'page',
        })

        if content_mode == "hybrid":
            if "content" not in page:
                raise ValueError(
                    "Manifest has no extracted content. "
                    "Rerun prepare_pdf.py --extract-content."
                )
            chunks.append(
                f"\nPage {number} Extracted content:\n"
            )
            for block in page["content"]:
                if block["type"] == "text":
                    chunks.append(
                        block["text"] + "\n"
                    )

                elif (
                    block["type"] == "image"
                    and inline_images
                ):
                    chunks.append(
                        "[Image region from PDF page]\n"
                        "<image>\n"
                    )
                    image_items.append({
                        'path': block['path'],
                        'type': 'crop',
                    })

    chunks.append(
        "\n###End of the PDF\n\n"
        f"{question}\n"
    )

    prompt = "".join(chunks)

    assert prompt.count('<image>') == len(image_items)
    return prompt, image_items


def prompt_to_messages(prompt, images):
    parts = [prompt]
    if images is not None and len(images) > 0:
        parts = prompt.split("<image>")

    if len(parts) - 1 != len(images):
        raise ValueError("Placeholder count does not match image count")

    content = []
    for i, text in enumerate(parts):
        if text:
            content.append(
                {
                    "type": "text",
                    "text": text,
                }
            )

        if i < len(images):
            content.append(
                {
                    "type": "image",
                    "image": images[i],
                }
            )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def resize_for_training(images, max_edge=512):
    out = []
    for im in images:
        im = im.copy()
        im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        out.append(im)
    return out


class PDFQuestionDataset(Dataset):
    def __init__(
        self,
        records,
        page_mode="selected",
        content_mode="hybrid",
        inline_images=True,
        questions_path=None,
        database_root=None,
        use_pdf=True,
        standard_only=False,
    ):
        if page_mode not in {"selected", "all"}:
            raise ValueError("page_mode must be selected or all")

        self.records = records
        self.page_mode = page_mode
        self.content_mode = content_mode
        self.inline_images = inline_images

        # Useful when manifests in questions.json use relative paths.
        self.questions_dir = (
            Path(questions_path).resolve().parent
            if questions_path is not None
            else Path.cwd()
        )
        self.database_root = (
            Path(database_root).resolve()
            if database_root is not None
            else None
        )

        # Cache manifests so questions from the same PDF do not repeatedly
        # parse the manifest JSON.
        #
        # Each DataLoader worker gets its own cache.
        self.manifest_cache = {}
        self.use_pdf = use_pdf
        self.standard_only = standard_only

    def __len__(self):
        return len(self.records)

    def resolve_manifest_path(self, record):
        manifest_path = record.get("manifest")

        if not manifest_path:
            raise ValueError(f"Question {record.get('qid')} has no 'manifest' field")

        path = Path(manifest_path)

        if not path.is_absolute():
            path = self.questions_dir / path

        path = path.resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {path}"
            )

        return path

    def get_manifest(self, path):
        key = str(path)

        if key not in self.manifest_cache:
            self.manifest_cache[key] = read_manifest(path)

        return self.manifest_cache[key]

    def __getitem__(self, idx):
        record = self.records[idx]

        sample = {
            "record": record,
            "manifest_path": None,
            "pages": [],
            "prompt": None,
            "image_paths": [],
            "images": None,
            "error": None,
        }

        try:
            if "retrieved_pages" in record:
                # RAG mode: pages can come from multiple PDFs.
                if self.database_root is None:
                    raise ValueError("database_root is required for retrieval mode")

                pages = []
                seen = set()
                for item in record["retrieved_pages"]:
                    doc_id = item["doc_id"]

                    manifest_path = (self.database_root / doc_id / "manifest.json").resolve()

                    # Ensure retrieved paths stay within database_root.
                    if not manifest_path.is_relative_to(self.database_root):
                        raise ValueError(f"Invalid document ID: {doc_id}")

                    manifest = self.get_manifest(manifest_path)

                    pages_by_number = {int(p["page"]): p for p in manifest["pages"]}

                    for number in item["pages"]:
                        number = int(number)
                        key = (doc_id, number)

                        if key in seen:
                            continue

                        if number not in pages_by_number:
                            raise ValueError(f"Page {number} not found in {doc_id}")
                        seen.add(key)
                        # Avoid modifying cached manifest data.
                        page = dict(pages_by_number[number])
                        page["_doc_id"] = doc_id
                        pages.append(page)

                if not pages:
                    raise ValueError("No retrieved pages for question")
            elif self.use_pdf:
                manifest_path = self.resolve_manifest_path(record)
                manifest = self.get_manifest(manifest_path)

                pages_by_number = {int(page["page"]): page for page in manifest["pages"]}

                if self.page_mode == "all":
                    numbers = sorted(pages_by_number)
                else:
                    numbers = list(dict.fromkeys(map(int, record.get("pages", []))))

                if not numbers:
                    raise ValueError("No pages chosen for question")

                missing = [p for p in numbers if p not in pages_by_number]

                if missing:
                    raise ValueError(f"Pages not found in manifest: {missing}")

                pages = [pages_by_number[p] for p in numbers]

            images = []
            paths = []
            if self.use_pdf:
                prompt, image_items = make_pdf_prompt(
                    record['question'],
                    pages,
                    self.content_mode,
                    self.inline_images,
                    standard_only=self.standard_only
                )
                for item in image_items:
                    path = item['path']

                    with Image.open(path) as im:
                        image = im.convert('RGB')
                        # Keep whole PDF pages at original resolution. Only shrink cropped figures.
                        if item['type'] == 'crop':
                            image.thumbnail((224, 224), Image.Resampling.LANCZOS)
                        images.append(image.copy())
                        paths.append(path)
            else:
                prompt = record['question']

            sample.update(
                prompt=prompt,
                image_paths=paths,
                images=images
            )

        except Exception as exc:
            sample["error"] = (f"{type(exc).__name__}: {exc}")

        # sample["load_seconds"] = round(time.perf_counter() - start, 3)

        return sample


# -----------------------------------------------------------------------------
# Reconstruction of planner states for offline multi-hop preference training.
# The initial/decision prompts reproduce the templates from decide_next_hop().
# Document excerpts are exact ONLY if matched_children are logged per parent;
# PDF page IDs alone do not identify the reranker's selected child snippets.
# -----------------------------------------------------------------------------

def planner_first_prompt(question):
    return (
        "You are generating the query to search for information to answer an Engineering question to get "
        "information needed to answer a question accurately from documents. Account for applicable "
        "conditions, exceptions, referenced clauses, annexes and tables. "
        "Do not answer the question. "
        "Return ONLY one JSON object: "
        '{"query":"specific missing information, table or topic"}. '
        f"Question: {question}"
    )


def planner_decision_prompt(question, excerpts):
    return (
        "You are checking whether retrieved documents contain ALL "
        "information needed to answer a question accurately. Account for applicable "
        "conditions, exceptions, referenced clauses, annexes and tables. "
        "Do not answer the engineering question. "
        "Return ONLY one JSON object: "
        '{"decision":"ANSWER"} or '
        '{"decision":"RETRIEVE","query":"specific missing clause, table or topic"}. '
        "Choose RETRIEVE if an important referenced requirement is missing. "
        "Choose ANSWER if the available excerpts are sufficient or if another "
        "retrieval is not needed.\n\n"
        f"Question: {question}\n\nEvidence:\n" + "\n\n".join(excerpts)
    )


def _planner_parent_excerpt(dataset, parent):
    """Return parent excerpt, using exact saved matched children if present.

    If no child snippets were logged, approximate the original retriever evidence
    with the first 1800 characters of manifest-extracted parent-page text.
    """
    doc_id = parent["doc_id"]
    pages = parent["pages"]
    children = parent.get("matched_children")
    if children is not None:
        excerpt = "\n".join(
            str(child["text"])[:1200] for child in children[:2]
        )
    else:
        if dataset.database_root is None:
            raise ValueError("database_root is needed to reconstruct planner excerpts")
        manifest_path = (dataset.database_root / doc_id / "manifest.json").resolve()
        if not manifest_path.is_relative_to(dataset.database_root):
            raise ValueError(f"Invalid doc_id: {doc_id}")
        manifest = dataset.get_manifest(manifest_path)
        by_number = {int(page["page"]): page for page in manifest["pages"]}
        texts = []
        for number in pages:
            page = by_number[int(number)]
            texts.extend(block.get("text", "") for block in page.get("content", [])
                         if block.get("type") == "text")
        excerpt = "\n".join(texts)[:1800]
    return f"Document {doc_id}, pages {pages}:\n{excerpt}"


def reconstruct_planner_segments(dataset, record):
    """Return (prompt, generated_action_json) for actual planner calls.

    Each hop must have CUMULATIVE `retrieved_pages` in original parent order.
    The query for hop t is generated by the preceding planner call.
    Never synthesize an ANSWER call when termination was due to hop/page limits.
    """
    question = record["question"]
    trace = record.get("retrieval_trace") or []
    segments = []
    for i, hop in enumerate(trace):
        decision = hop.get("decision")
        if decision is None:
            continue
        decision = str(decision).upper()
        if decision not in {"ANSWER", "RETRIEVE"}:
            raise ValueError(f"Invalid decision at hop {i+1}: {decision}")
        if "retrieved_pages" not in hop:
            raise ValueError(f"Missing cumulative retrieved_pages for hop {i+1}")
        excerpts = [_planner_parent_excerpt(dataset, parent)
                    for parent in hop["retrieved_pages"]]
        prompt = planner_decision_prompt(question, excerpts)
        if decision == "RETRIEVE":
            action = {"decision": "RETRIEVE", "query": hop["query"]}
        else:
            action = {"decision": "ANSWER"}
        segments.append((prompt, json.dumps(action, ensure_ascii=False)))
    return segments

