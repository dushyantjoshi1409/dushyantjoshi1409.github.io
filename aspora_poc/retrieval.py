"""
Hybrid Retrieval Engine — the "Two Detectives" from your interview answer.

Detective 1: Semantic search (TF-IDF cosine similarity as a local proxy for Pinecone embeddings)
Detective 2: BM25 keyword search (exact match, good for tickers and precise terms)
Fusion: Reciprocal Rank Fusion (RRF) merges both ranked lists
Reranker: Simple keyword-overlap reranker as a proxy for cross-encoder

In production at Aspora:
  - Detective 1 = Pinecone with text-embedding-3-small
  - Detective 2 = Elasticsearch with BM25
  - Fusion = same RRF formula
  - Reranker = Cohere rerank or cross-encoder model
"""

import math
import re
from collections import Counter
from typing import Optional

from rank_bm25 import BM25Okapi

from data.sample_data import DOCUMENTS


# -------------------------------------------------------
# 1. DOCUMENT INDEX — built once at startup (like Singleton at Leaps)
# -------------------------------------------------------
class DocumentIndex:
    """Indexes all documents for both BM25 and TF-IDF search."""

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.tokenized = [self._tokenize(d["text"]) for d in documents]

        # BM25 index (Detective 2 — keyword search)
        self.bm25 = BM25Okapi(self.tokenized)

        # TF-IDF vectors (Detective 1 — semantic proxy)
        self.idf = self._compute_idf()
        self.tfidf_vectors = [self._tfidf_vector(tokens) for tokens in self.tokenized]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())

    def _compute_idf(self) -> dict[str, float]:
        n = len(self.documents)
        df = Counter()
        for tokens in self.tokenized:
            for t in set(tokens):
                df[t] += 1
        return {t: math.log((n + 1) / (freq + 1)) + 1 for t, freq in df.items()}

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        total = len(tokens) or 1
        return {t: (count / total) * self.idf.get(t, 0) for t, count in tf.items()}


# -------------------------------------------------------
# 2. METADATA FILTER — narrows search BEFORE retrieval
#    "Reliance in Q2 2024" → filter to ticker=RELIANCE, date=2024-Q2
#    Massively improves precision and speed
# -------------------------------------------------------
def filter_by_metadata(
    documents: list[dict],
    ticker: Optional[str] = None,
    date: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> list[dict]:
    """Filter documents by metadata before doing semantic/keyword search."""
    filtered = documents
    if ticker:
        filtered = [d for d in filtered if d["metadata"].get("ticker") == ticker.upper()
                     or d["metadata"].get("company", "").upper() == ticker.upper()]
    if date:
        filtered = [d for d in filtered if date in d["metadata"].get("date", "")]
    if doc_type:
        filtered = [d for d in filtered if d["metadata"].get("type") == doc_type]
    return filtered


# -------------------------------------------------------
# 3. SEMANTIC SEARCH (Detective 1) — TF-IDF cosine similarity
#    In production: Pinecone with OpenAI embeddings
#    Here: TF-IDF as a lightweight local proxy
# -------------------------------------------------------
def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values())) or 1
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values())) or 1
    return dot / (mag_a * mag_b)


def semantic_search(query: str, index: DocumentIndex, top_k: int = 5,
                    candidate_docs: Optional[list[dict]] = None) -> list[tuple[dict, float]]:
    """
    Detective 1: Find documents by MEANING.
    Returns: list of (document, score) sorted by relevance.
    """
    query_tokens = index._tokenize(query)
    query_vec = index._tfidf_vector(query_tokens)

    candidates = candidate_docs if candidate_docs else index.documents
    results = []
    for doc in candidates:
        idx = next(i for i, d in enumerate(index.documents) if d["id"] == doc["id"])
        score = cosine_similarity(query_vec, index.tfidf_vectors[idx])
        results.append((doc, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# -------------------------------------------------------
# 4. BM25 KEYWORD SEARCH (Detective 2)
#    In production: Elasticsearch
#    Here: rank-bm25 library (same algorithm)
# -------------------------------------------------------
def keyword_search(query: str, index: DocumentIndex, top_k: int = 5,
                   candidate_docs: Optional[list[dict]] = None) -> list[tuple[dict, float]]:
    """
    Detective 2: Find documents by EXACT WORDS.
    BM25 is great for: ticker symbols (NVDA), exact financial terms, precise names.
    """
    query_tokens = index._tokenize(query)

    if candidate_docs:
        # Build a temporary BM25 index over filtered candidates
        candidate_tokenized = [index._tokenize(d["text"]) for d in candidate_docs]
        if not candidate_tokenized:
            return []
        temp_bm25 = BM25Okapi(candidate_tokenized)
        scores = temp_bm25.get_scores(query_tokens)
        results = [(candidate_docs[i], scores[i]) for i in range(len(candidate_docs))]
    else:
        scores = index.bm25.get_scores(query_tokens)
        results = [(index.documents[i], scores[i]) for i in range(len(index.documents))]

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# -------------------------------------------------------
# 5. RECIPROCAL RANK FUSION (RRF) — merges two detective lists
#    Formula: score = sum( 1 / (k + rank) ) for each retriever
#    k = 60 is standard (prevents top-1 from dominating)
# -------------------------------------------------------
def reciprocal_rank_fusion(
    semantic_results: list[tuple[dict, float]],
    keyword_results: list[tuple[dict, float]],
    k: int = 60,
) -> list[tuple[dict, float]]:
    """
    The police chief merging both detectives' suspect lists.
    A document ranked high by BOTH detectives gets the best combined score.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    # Score from Detective 1 (semantic)
    for rank, (doc, _) in enumerate(semantic_results, start=1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
        doc_map[doc_id] = doc

    # Score from Detective 2 (keyword)
    for rank, (doc, _) in enumerate(keyword_results, start=1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
        doc_map[doc_id] = doc

    # Sort by combined RRF score
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[doc_id], score) for doc_id, score in ranked]


# -------------------------------------------------------
# 6. SIMPLE RERANKER — proxy for cross-encoder
#    In production: Cohere rerank or cross-encoder model
#    Here: keyword overlap + exact match bonus
# -------------------------------------------------------
def simple_rerank(query: str, results: list[tuple[dict, float]], top_k: int = 3) -> list[tuple[dict, float]]:
    """
    The expert second opinion.
    In production: cross-encoder takes (query, document) as pair input.
    Here: keyword overlap as a simple proxy.
    """
    query_words = set(re.findall(r'\w+', query.lower()))
    reranked = []
    for doc, rrf_score in results:
        doc_words = set(re.findall(r'\w+', doc["text"].lower()))
        overlap = len(query_words & doc_words) / (len(query_words) or 1)
        # Combine RRF score with overlap score
        combined = 0.6 * rrf_score + 0.4 * overlap
        reranked.append((doc, combined))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


# -------------------------------------------------------
# 7. FULL HYBRID RETRIEVAL PIPELINE
#    This is what the orchestrator calls
# -------------------------------------------------------
def hybrid_retrieve(
    query: str,
    index: DocumentIndex,
    ticker: Optional[str] = None,
    date: Optional[str] = None,
    doc_type: Optional[str] = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Complete retrieval pipeline:
    1. Metadata filter (narrow the search space)
    2. Semantic search (Detective 1 — meaning)
    3. Keyword search (Detective 2 — exact match)
    4. RRF fusion (merge both lists)
    5. Rerank (expert second opinion)
    6. Return top-k documents with scores
    """
    # Step 1: Metadata filtering
    candidates = filter_by_metadata(DOCUMENTS, ticker=ticker, date=date, doc_type=doc_type)
    if not candidates:
        candidates = DOCUMENTS  # fallback to full corpus

    # Step 2 & 3: Both detectives search
    sem_results = semantic_search(query, index, top_k=10, candidate_docs=candidates)
    kw_results = keyword_search(query, index, top_k=10, candidate_docs=candidates)

    # Step 4: RRF fusion
    fused = reciprocal_rank_fusion(sem_results, kw_results)

    # Step 5: Rerank
    reranked = simple_rerank(query, fused, top_k=top_k)

    # Return documents with retrieval metadata
    return [
        {
            "document": doc,
            "score": score,
            "text": doc["text"],
            "source": f"{doc['metadata']['company']} - {doc['metadata']['date']} ({doc['metadata']['type']})",
        }
        for doc, score in reranked
    ]


# Build the global index (Singleton pattern — same as Leaps)
INDEX = DocumentIndex(DOCUMENTS)
