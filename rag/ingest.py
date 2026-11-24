"""Turn the guidelines PDF into clean, page-tagged chunks and store them in ChromaDB.

Fixes compared to the first version of the notebook:

1. Clean text: running titles, page numbers and "Chapter N" lines are removed,
   and pages are joined with a line break, so they are not glued together
   ("2025In Memoriam") or injected into the middle of sentences.
2. No reference lists: the bibliography at the end of each chapter is dropped,
   so citations do not compete with real content during search.
3. Page metadata: every chunk remembers its page(s) and section, so sources
   can be shown by the code instead of guessed by the model.
4. Safe to re-run: the collection is rebuilt from scratch with stable ids,
   so running ingestion twice no longer duplicates every chunk.
"""
from __future__ import annotations

import hashlib
import re
from bisect import bisect_right
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag import config

_CITATION = re.compile(r"et al|doi|\b(19|20)\d{2};\s?\d")


@dataclass
class Page:
    number: int  # 1-based page number as printed in the PDF viewer
    section: str  # e.g. "SCREENING AND PREVENTION", "" if unknown
    text: str


def _strip_page_furniture(raw: str, number: int) -> tuple[str, str | None]:
    """Remove the running title, page number and chapter label at the top of a page.

    Even pages start with: title / page number / SECTION NAME.
    Odd pages start with:  page number / Chapter N.
    Returns the cleaned text and the section name if the page printed one.
    """
    lines = [line.rstrip() for line in raw.splitlines()]
    section = None
    had_title = False
    while lines:
        head = lines[0].strip()
        if head == config.RUNNING_TITLE:
            had_title = True
        elif head == str(number) or re.fullmatch(r"Chapter \d+", head):
            pass
        elif had_title and section is None and head.isupper() and len(head) < 80:
            section = head  # the section heading printed under the title
        else:
            break
        lines.pop(0)
    return "\n".join(lines).strip(), section


def load_pages(pdf_path=config.PDF_PATH) -> list[Page]:
    """Extract clean page texts, skipping the reference lists."""
    reader = PdfReader(str(pdf_path))
    pages: list[Page] = []
    section = ""
    in_references = False

    for index, pdf_page in enumerate(reader.pages):
        number = index + 1
        text, printed_section = _strip_page_furniture(pdf_page.extract_text() or "", number)
        if printed_section:
            section = printed_section

        # A "REFERENCES" heading starts the bibliography: keep only what comes before it.
        match = re.search(r"^\s*REFERENCES\s*$", text, flags=re.MULTILINE)
        if match:
            text = text[: match.start()]
            in_references = True
        elif in_references:
            # Following pages stay in the bibliography while they are dense with citations.
            if len(_CITATION.findall(text)) >= 4:
                continue
            in_references = False

        if text.strip():
            pages.append(Page(number=number, section=section, text=text.strip()))
    return pages


def make_chunks(pages: list[Page], size: int | None = None, overlap: int | None = None) -> list[Document]:
    """Split the cleaned text into overlapping chunks tagged with page and section."""
    # Join pages with a blank line and remember where each page starts in the joined text.
    starts, parts, offset = [], [], 0
    for page in pages:
        starts.append(offset)
        parts.append(page.text)
        offset += len(page.text) + 2
    full_text = "\n\n".join(parts)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size or config.CHUNK_SIZE,
        chunk_overlap=overlap if overlap is not None else config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
        add_start_index=True,
    )
    documents = []
    for doc in splitter.create_documents([full_text]):
        start = doc.metadata["start_index"]
        first = pages[bisect_right(starts, start) - 1]
        last = pages[bisect_right(starts, start + len(doc.page_content) - 1) - 1]
        doc.metadata = {
            "page_start": first.number,
            "page_end": last.number,
            "section": first.section,
        }
        documents.append(doc)
    return documents


def chunk_id(doc: Document, position: int) -> str:
    """Stable id: the same chunk always gets the same id."""
    digest = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{doc.metadata['page_start']}-{position}-{digest}"


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": config.QUERY_PREFIX},
    )


def build_index(documents: list[Document], embeddings=None, persist: bool = True):
    """Create the vector store from scratch. Running it twice gives the same result."""
    from langchain_chroma import Chroma

    embeddings = embeddings or get_embeddings()
    kwargs = {"collection_name": config.COLLECTION, "embedding_function": embeddings}
    if persist:
        kwargs["persist_directory"] = str(config.CHROMA_DIR)

    store = Chroma(**kwargs)
    store.delete_collection()  # start clean instead of appending to old data
    store = Chroma(**kwargs)
    store.add_documents(documents, ids=[chunk_id(d, i) for i, d in enumerate(documents)])
    return store


def load_index(embeddings=None):
    """Open the persisted vector store without re-embedding anything."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


if __name__ == "__main__":
    pages = load_pages()
    chunks = make_chunks(pages)
    store = build_index(chunks)
    print(f"Pages kept: {len(pages)} | chunks: {len(chunks)} | stored: {store._collection.count()}")
