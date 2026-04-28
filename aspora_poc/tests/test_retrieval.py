"""Tests for hybrid retrieval engine — FAISS + BM25 + RRF + reranking."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from data.ingest import get_datastore
from engines.retrieval import (
    hybrid_retrieve, faiss_search, bm25_search,
    reciprocal_rank_fusion, filter_by_metadata,
)


@pytest.fixture(scope="module")
def store():
    return get_datastore()


class TestFAISSSearch:
    def test_returns_results(self, store):
        results = faiss_search("NVIDIA data center revenue", store, top_k=3)
        assert len(results) > 0
        assert all(isinstance(score, float) for _, score in results)

    def test_nvidia_query_returns_nvidia_docs(self, store):
        results = faiss_search("NVIDIA data center revenue Q3", store, top_k=3)
        tickers = [doc["metadata"]["ticker"] for doc, _ in results]
        assert "NVDA" in tickers

    def test_top_k_limit(self, store):
        results = faiss_search("revenue growth", store, top_k=2)
        assert len(results) <= 2


class TestBM25Search:
    def test_returns_results(self, store):
        results = bm25_search("NVIDIA data center revenue", store, top_k=3)
        assert len(results) > 0

    def test_exact_term_matching(self, store):
        results = bm25_search("NVIDIA data center", store, top_k=3)
        # BM25 should find docs with exact terms
        top_doc = results[0][0] if results else None
        assert top_doc is not None
        assert "NVIDIA" in top_doc["text"]


class TestRRFFusion:
    def test_combines_results(self, store):
        sem = faiss_search("NVIDIA revenue", store, top_k=5)
        kw = bm25_search("NVIDIA revenue", store, top_k=5)
        fused = reciprocal_rank_fusion(sem, kw)
        assert len(fused) > 0
        # Fused should contain docs from both
        doc_ids = [doc["id"] for doc, _ in fused]
        assert len(set(doc_ids)) == len(doc_ids)  # no duplicates


class TestMetadataFilter:
    def test_filter_by_ticker(self, store):
        filtered, indices = filter_by_metadata(store.documents, ticker="NVDA")
        assert len(filtered) > 0
        assert all(d["metadata"]["ticker"] == "NVDA" for d in filtered)

    def test_filter_by_date(self, store):
        filtered, indices = filter_by_metadata(store.documents, date="2024-Q3")
        assert len(filtered) > 0
        assert all("2024-Q3" in d["metadata"]["date"] for d in filtered)

    def test_no_match_returns_empty(self, store):
        filtered, indices = filter_by_metadata(store.documents, ticker="DOESNOTEXIST")
        assert len(filtered) == 0


class TestHybridRetrieval:
    def test_full_pipeline(self, store):
        results = hybrid_retrieve("What was NVIDIA's data center revenue?", store)
        assert len(results) > 0
        assert all("text" in r and "source" in r and "score" in r for r in results)

    def test_with_ticker_filter(self, store):
        results = hybrid_retrieve("revenue", store, ticker="TCS")
        assert len(results) > 0
        # Should prioritize TCS docs
        assert any("TCS" in r["source"] for r in results)

    def test_compliance_docs(self, store):
        results = hybrid_retrieve("NRI tax DTAA UAE", store)
        assert len(results) > 0
        # Should find compliance guides
        assert any("compliance" in r["source"].lower() or "REGULATORY" in r["source"] for r in results)
