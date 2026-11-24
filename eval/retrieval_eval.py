"""Retrieval evaluation: does the top-k contain a page that actually holds the answer?

Compares four pipelines on the same questions, one change at a time:
  original     the first notebook as committed (raw text, chunks stored 3 times)
  no_dupes     the same, but each chunk stored once (fixes the duplicate bug)
  cleaned      clean text, no reference lists, page metadata (same model and chunk size)
  final        the current rag/ module: bge-base embeddings + BM25 hybrid, 1000-char chunks

Metrics (each question has an "evidence" regex that matches the answer text):
  Hit@k        a retrieved chunk actually contains the evidence (strict, the main metric)
  page Hit@k   a retrieved chunk comes from a page that contains the evidence (lenient;
               can be a lucky hit from an unrelated chunk on a busy page)
  rank         position of the first chunk containing the evidence, searched to depth 50

Run:  python -m eval.retrieval_eval
No API key needed: this only tests the search step, not Claude.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_right
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag import config
from langchain_huggingface import HuggingFaceEmbeddings

from rag.ingest import get_embeddings, load_pages, make_chunks
from rag.retrieve import HybridRetriever

HERE = Path(__file__).parent


def gold_pages(questions, pages):
    """Map each question id to the content pages that contain its evidence."""
    norm = {p.number: re.sub(r"\s+", " ", p.text) for p in pages}
    gold = {}
    for q in questions:
        pattern = re.compile(q["evidence"], re.S)
        gold[q["id"]] = {n for n, text in norm.items() if pattern.search(text)}
        assert gold[q["id"]], f"no page matches the evidence for {q['id']}"
    return gold


def original_chunks():
    """Rebuild the first notebook's chunks exactly, with the page each one starts on."""
    raw_pages = [p.extract_text() or "" for p in PdfReader(str(config.PDF_PATH)).pages]
    starts, offset = [], 0
    for text in raw_pages:
        starts.append(offset)
        offset += len(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "], add_start_index=True
    )
    docs = splitter.create_documents(["".join(raw_pages)])
    for doc in docs:
        start = doc.metadata["start_index"]
        end = start + len(doc.page_content) - 1
        doc.metadata = {"page_start": bisect_right(starts, start), "page_end": bisect_right(starts, end)}
    return docs


DEPTH = 50


def contains_evidence(doc, pattern):
    return bool(pattern.search(re.sub(r"\s+", " ", doc.page_content)))


def evaluate(name, search, questions, gold, k=config.TOP_K):
    hits = page_hits = distinct_total = 0
    rows = []
    for q in questions:
        pattern = re.compile(q["evidence"], re.S)
        results = search(q["question"], DEPTH)
        top = results[:k]
        hit = any(contains_evidence(d, pattern) for d in top)
        page_hit = any(
            any(p in gold[q["id"]] for p in range(d.metadata["page_start"], d.metadata["page_end"] + 1))
            for d in top
        )
        rank = next((i + 1 for i, d in enumerate(results) if contains_evidence(d, pattern)), None)
        distinct = len({d.page_content for d in top})
        hits += hit
        page_hits += page_hit
        distinct_total += distinct
        rows.append({"id": q["id"], "hit": hit, "page_hit": page_hit, "rank": rank, "distinct": distinct,
                     "pages": [d.metadata["page_start"] for d in top]})
    n = len(questions)
    ranks = [r["rank"] for r in rows if r["rank"]]
    summary = {"pipeline": name, "hits": hits, "page_hits": page_hits, "questions": n,
               "hit_at_k": round(hits / n, 3), "page_hit_at_k": round(page_hits / n, 3),
               "found_in_top_50": len(ranks), "median_rank": sorted(ranks)[len(ranks) // 2] if ranks else None,
               "avg_distinct_chunks": round(distinct_total / n, 2), "k": k}
    return summary, rows


def main():
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))["questions"]
    pages = load_pages()
    gold = gold_pages(questions, pages)
    minilm = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # the original embedding model
    orig = original_chunks()
    variants = {
        # name: (chunks, embeddings, hybrid search?)
        "original": (orig * 3, minilm, False),  # the notebook ran ingestion three times
        "no_dupes": (orig, minilm, False),
        "cleaned": (make_chunks(pages, size=500, overlap=50), minilm, False),
        "final": (make_chunks(pages), get_embeddings(), True),
    }

    report = {"k": config.TOP_K, "summaries": [], "details": {}, "gold_pages": {k: sorted(v) for k, v in gold.items()}}
    for name, (docs, embeddings, hybrid) in variants.items():
        store = Chroma(collection_name=f"eval_{name}", embedding_function=embeddings)
        store.add_documents(docs, ids=[f"{name}-{i}" for i in range(len(docs))])
        if hybrid:
            retriever = HybridRetriever(store)
            search = lambda question, depth: retriever.search(question, k=depth)
        else:
            search = lambda question, depth: store.similarity_search(question, k=depth)
        summary, rows = evaluate(name, search, questions, gold)
        summary["chunks_stored"] = len(docs)
        report["summaries"].append(summary)
        report["details"][name] = rows
        store.delete_collection()

    out = HERE / "results" / "retrieval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    k = config.TOP_K
    print(f"{'pipeline':<16} {'chunks':>7} {f'Hit@{k}':>8} {f'page Hit@{k}':>12} {'in top 50':>10} {'median rank':>12} {'distinct/5':>11}")
    for s in report["summaries"]:
        print(f"{s['pipeline']:<16} {s['chunks_stored']:>7} {s['hits']:>5}/{s['questions']:<2} {s['page_hits']:>9}/{s['questions']:<2}"
              f" {s['found_in_top_50']:>7}/{s['questions']:<2} {str(s['median_rank']):>12} {s['avg_distinct_chunks']:>11}")
    print(f"\nRank of the first chunk that contains the answer (- = not in top {DEPTH}; * = inside top {k}):")
    names = list(variants)
    print(f"{'question':<18}" + "".join(f"{n:>16}" for n in names))
    for i, q in enumerate(questions):
        cells = []
        for n in names:
            r = report["details"][n][i]["rank"]
            cells.append(f"{'-' if r is None else str(r) + ('*' if r <= k else ' '):>16}")
        print(f"{q['id']:<18}" + "".join(cells))
    print(f"\nSaved to {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
