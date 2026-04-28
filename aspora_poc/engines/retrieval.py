"""
Hybrid Retrieval Engine — FAISS (semantic) + BM25 (keyword) + RRF fusion + reranker.

Upgraded from TF-IDF to real FAISS vector search with sentence-transformer embeddings.
BM25 and RRF fusion logic preserved from original skeleton.
Reranker uses sentence-transformer cross-similarity instead of keyword overlap.
"""

import logging
import re
import time
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from config.settings import (
    RRF_K, FAISS_TOP_K, BM25_TOP_K, RERANKER_TOP_K,
)
from data.ingest import DataStore, get_datastore

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# 1. METADATA FILTER — narrows search BEFORE retrieval
# -------------------------------------------------------
def filter_by_metadata(
    documents: list[dict],
    ticker: Optional[str] = None,
    date: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> tuple[list[dict], list[int]]:
    """
    Filter documents by metadata before semantic/keyword search.
    Returns (filtered_docs, original_indices) so we can map back to FAISS IDs.
    """
    results = []
    indices = []
    for i, doc in enumerate(documents):
        meta = doc.get("metadata", {})
        if ticker:
            if (meta.get("ticker", "").upper() != ticker.upper() and
                    meta.get("company", "").upper() != ticker.upper()):
                continue
        if date:
            if date not in meta.get("date", ""):
                continue
        if doc_type:
            if meta.get("type") != doc_type:
                continue
        results.append(doc)
        indices.append(i)
    return results, indices


# -------------------------------------------------------
# 2. FAISS SEMANTIC SEARCH (Detective 1)
#    Real embeddings → real vector similarity
# -------------------------------------------------------
def faiss_search(
    query: str,
    store: DataStore,
    top_k: int = FAISS_TOP_K,
    candidate_indices: Optional[list[int]] = None,
) -> list[tuple[dict, float]]:
    """
    Detective 1: Find documents by MEANING using FAISS.
    Uses sentence-transformer embeddings for real semantic search.
    """
    start = time.time()
    query_embedding = store.embed_query(query)

    if candidate_indices is not None and len(candidate_indices) > 0:
        # Search only within filtered candidates by building candidate vectors
        candidate_embeddings = np.array(
            [store.embeddings[i] for i in candidate_indices], dtype=np.float32
        )
        # Compute inner product directly
        scores = np.dot(candidate_embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            (store.documents[candidate_indices[i]], float(scores[i]))
            for i in top_indices
        ]
    else:
        # Full FAISS index search
        scores, indices = store.faiss_index.search(query_embedding, top_k)
        results = [
            (store.documents[idx], float(scores[0][i]))
            for i, idx in enumerate(indices[0])
            if idx >= 0  # FAISS returns -1 for padding
        ]

    elapsed = round((time.time() - start) * 1000, 1)
    logger.debug(f"FAISS search: {len(results)} results in {elapsed}ms")
    return results


# -------------------------------------------------------
# 3. BM25 KEYWORD SEARCH (Detective 2)
# -------------------------------------------------------
def bm25_search(
    query: str,
    store: DataStore,
    top_k: int = BM25_TOP_K,
    candidate_indices: Optional[list[int]] = None,
) -> list[tuple[dict, float]]:
    """
    Detective 2: Find documents by EXACT WORDS.
    BM25 excels at: ticker symbols, exact financial terms, precise names.
    """
    start = time.time()
    query_tokens = store._tokenize(query)

    if candidate_indices is not None and len(candidate_indices) > 0:
        # Build temporary BM25 over filtered candidates
        candidate_tokenized = [store.tokenized_docs[i] for i in candidate_indices]
        if not candidate_tokenized:
            return []
        temp_bm25 = BM25Okapi(candidate_tokenized)
        scores = temp_bm25.get_scores(query_tokens)
        results = [
            (store.documents[candidate_indices[i]], float(scores[i]))
            for i in range(len(candidate_indices))
        ]
    else:
        scores = store.bm25_index.get_scores(query_tokens)
        results = [
            (store.documents[i], float(scores[i]))
            for i in range(len(store.documents))
        ]

    results.sort(key=lambda x: x[1], reverse=True)
    elapsed = round((time.time() - start) * 1000, 1)
    logger.debug(f"BM25 search: {len(results[:top_k])} results in {elapsed}ms")
    return results[:top_k]


# -------------------------------------------------------
# 4. RECIPROCAL RANK FUSION (RRF)
# -------------------------------------------------------
def reciprocal_rank_fusion(
    semantic_results: list[tuple[dict, float]],
    keyword_results: list[tuple[dict, float]],
    k: int = RRF_K,
) -> list[tuple[dict, float]]:
    """
    Merge both detectives' ranked lists.
    A document ranked high by BOTH gets the best combined score.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, (doc, _) in enumerate(semantic_results, start=1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
        doc_map[doc_id] = doc

    for rank, (doc, _) in enumerate(keyword_results, start=1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
        doc_map[doc_id] = doc

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[doc_id], score) for doc_id, score in ranked]


# -------------------------------------------------------
# 5. RERANKER — sentence-transformer cross-similarity
# -------------------------------------------------------
def rerank(
    query: str,
    results: list[tuple[dict, float]],
    store: DataStore,
    top_k: int = RERANKER_TOP_K,
) -> list[tuple[dict, float]]:
    """
    Reranker: encode query and each candidate document, compute similarity.
    Better than keyword overlap because it captures semantic meaning.
    """
    if not results:
        return []

    start = time.time()
    query_emb = store.embedding_model.encode([query], normalize_embeddings=True)
    doc_texts = [doc["text"] for doc, _ in results]
    doc_embs = store.embedding_model.encode(doc_texts, normalize_embeddings=True)

    # Cosine similarity (already normalized → dot product)
    scores = np.dot(doc_embs, query_emb.T).flatten()

    reranked = [
        (results[i][0], float(0.5 * results[i][1] + 0.5 * scores[i]))
        for i in range(len(results))
    ]
    reranked.sort(key=lambda x: x[1], reverse=True)

    elapsed = round((time.time() - start) * 1000, 1)
    logger.debug(f"Reranking: {len(reranked[:top_k])} results in {elapsed}ms")
    return reranked[:top_k]


# -------------------------------------------------------
# 6. FULL HYBRID RETRIEVAL PIPELINE
# -------------------------------------------------------
def hybrid_retrieve(
    query: str,
    store: Optional[DataStore] = None,
    ticker: Optional[str] = None,
    date: Optional[str] = None,
    doc_type: Optional[str] = None,
    top_k: int = RERANKER_TOP_K,
) -> list[dict]:
    """
    Complete retrieval pipeline:
    1. Metadata filter (narrow the search space)
    2. FAISS semantic search (Detective 1 — meaning)
    3. BM25 keyword search (Detective 2 — exact match)
    4. RRF fusion (merge both lists)
    5. Rerank (sentence-transformer cross-similarity)
    6. Return top-k documents with scores
    """
    if store is None:
        store = get_datastore()

    pipeline_start = time.time()

    # Step 1: Metadata filtering
    candidates, candidate_indices = filter_by_metadata(
        store.documents, ticker=ticker, date=date, doc_type=doc_type,
    )
    if not candidates:
        # Fallback to full corpus if no metadata matches
        candidates = store.documents
        candidate_indices = None
    else:
        logger.info(f"Metadata filter: {len(candidates)} candidates "
                     f"(ticker={ticker}, date={date})")

    # Step 2: FAISS semantic search
    sem_results = faiss_search(query, store, top_k=FAISS_TOP_K,
                                candidate_indices=candidate_indices)

    # Step 3: BM25 keyword search
    kw_results = bm25_search(query, store, top_k=BM25_TOP_K,
                              candidate_indices=candidate_indices)

    # Step 4: RRF fusion
    fused = reciprocal_rank_fusion(sem_results, kw_results)

    # Step 5: Rerank
    reranked = rerank(query, fused, store, top_k=top_k)

    elapsed = round((time.time() - pipeline_start) * 1000, 1)
    logger.info(f"Hybrid retrieval: {len(reranked)} docs in {elapsed}ms")

    # Return documents with retrieval metadata
    return [
        {
            "document": doc,
            "score": score,
            "text": doc["text"],
            "source": (f"{doc['metadata']['company']} - "
                       f"{doc['metadata']['date']} ({doc['metadata']['type']})"),
        }
        for doc, score in reranked
    ]
