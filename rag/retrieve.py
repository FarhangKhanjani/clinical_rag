"""Hybrid search: meaning-based (embeddings) + keyword-based (BM25), fused by rank.

Why both? Embeddings find text with the same meaning in different words
("first-line drug" vs "initial therapy"). Keyword search catches exact clinical
terms that embeddings tend to blur ("HbA1c", "FIB-4", "UACR"). Reciprocal rank
fusion (RRF) combines the two ranked lists without having to compare their scores.
"""
from __future__ import annotations

import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag import config


def tokenize(text: str) -> list[str]:
    """Lowercase words, keeping terms like "hba1c", "glp-1ra" and "6.5" intact."""
    return re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())


class HybridRetriever:
    def __init__(self, store):
        self.store = store
        data = store.get(include=["documents", "metadatas"])
        self.docs = [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs])

    def search(self, question: str, k: int = config.TOP_K) -> list[Document]:
        depth = min(config.CANDIDATES, len(self.docs))

        vector_hits = self.store.similarity_search(question, k=depth)
        keyword_scores = self.bm25.get_scores(tokenize(question))
        keyword_hits = [self.docs[i] for i in sorted(range(len(self.docs)), key=lambda i: -keyword_scores[i])[:depth]]

        # Reciprocal rank fusion: each list adds 1 / (RRF_K + rank) to a chunk's score.
        fused: dict[str, float] = {}
        by_text: dict[str, Document] = {}
        for ranked in (vector_hits, keyword_hits):
            for rank, doc in enumerate(ranked, start=1):
                fused[doc.page_content] = fused.get(doc.page_content, 0.0) + 1 / (config.RRF_K + rank)
                by_text[doc.page_content] = doc
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [by_text[text] for text in best]
