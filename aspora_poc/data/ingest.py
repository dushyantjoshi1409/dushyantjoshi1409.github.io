"""
Ingestion Pipeline — loads sample data, builds all indexes.
Runs once at startup (Singleton pattern).

Pipeline steps:
  1. Load documents from sample_data
  2. Generate embeddings using sentence-transformers (all-MiniLM-L6-v2)
  3. Build FAISS index (IndexFlatIP for cosine similarity)
  4. Build BM25 index from tokenized documents
  5. Build NetworkX graph from relationship data
  6. Load structured data into SQLite
"""

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

import faiss
import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSION, SQLITE_DB_PATH
from data.sample_data import (
    DOCUMENTS, STOCK_PRICES, FINANCIAL_METRICS, KNOWLEDGE_GRAPH, NRI_CONTEXTS,
)

logger = logging.getLogger(__name__)


class DataStore:
    """
    Singleton data store — all indexes built once, shared across the app.
    Holds: FAISS index, BM25 index, NetworkX graph, SQLite connection.
    """

    _instance: Optional["DataStore"] = None

    def __new__(cls) -> "DataStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        logger.info("🔧 Initializing DataStore — building all indexes...")
        start = time.time()

        self.documents = DOCUMENTS
        self.nri_contexts = NRI_CONTEXTS

        # Step 1: Load embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Step 2: Generate embeddings
        self._build_embeddings()

        # Step 3: Build FAISS index
        self._build_faiss_index()

        # Step 4: Build BM25 index
        self._build_bm25_index()

        # Step 5: Build NetworkX graph
        self._build_knowledge_graph()

        # Step 6: Load structured data into SQLite
        self._build_sqlite_db()

        elapsed = round((time.time() - start) * 1000, 1)
        logger.info(f"✅ DataStore initialized in {elapsed}ms — "
                     f"{len(self.documents)} docs, "
                     f"{self.faiss_index.ntotal} vectors, "
                     f"{self.graph.number_of_nodes()} graph nodes")

    # ----- Embeddings -----

    def _build_embeddings(self) -> None:
        """Generate embeddings for all documents using sentence-transformers."""
        texts = [doc["text"] for doc in self.documents]
        logger.info(f"Generating embeddings for {len(texts)} documents...")

        self.embeddings = self.embedding_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        # Store embeddings back on documents for reference
        for i, doc in enumerate(self.documents):
            doc["embedding"] = self.embeddings[i].tolist()

        logger.info(f"Embeddings shape: {self.embeddings.shape}")

    # ----- FAISS -----

    def _build_faiss_index(self) -> None:
        """Build FAISS IndexFlatIP for cosine similarity (normalized vectors → dot product = cosine)."""
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings.astype(np.float32))
        logger.info(f"FAISS index built: {self.faiss_index.ntotal} vectors, dim={dim}")

    # ----- BM25 -----

    def _build_bm25_index(self) -> None:
        """Build BM25 index from tokenized document texts."""
        self.tokenized_docs = [self._tokenize(doc["text"]) for doc in self.documents]
        self.bm25_index = BM25Okapi(self.tokenized_docs)
        logger.info(f"BM25 index built: {len(self.tokenized_docs)} documents")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r"\w+", text.lower())

    # ----- Knowledge Graph -----

    def _build_knowledge_graph(self) -> None:
        """Build NetworkX directed graph from sample KNOWLEDGE_GRAPH data."""
        self.graph = nx.MultiDiGraph()

        # Add nodes with attributes
        for node_id, attrs in KNOWLEDGE_GRAPH["nodes"].items():
            self.graph.add_node(node_id, **attrs)

        # Add edges with relation attribute
        for edge in KNOWLEDGE_GRAPH["edges"]:
            self.graph.add_edge(
                edge["from"], edge["to"], relation=edge["relation"],
            )
            # Add reverse edge for bidirectional relationships
            if edge["relation"] in ("COMPETES_WITH", "SAME_SECTOR"):
                self.graph.add_edge(
                    edge["to"], edge["from"], relation=edge["relation"],
                )

        logger.info(f"Knowledge graph built: {self.graph.number_of_nodes()} nodes, "
                     f"{self.graph.number_of_edges()} edges")

    # ----- SQLite -----

    def _build_sqlite_db(self) -> None:
        """Load structured stock data into SQLite — deterministic lookups, no LLM."""
        db_path = Path(SQLITE_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db_conn.row_factory = sqlite3.Row
        cursor = self.db_conn.cursor()

        # Stock prices table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                ticker TEXT PRIMARY KEY,
                price REAL,
                change TEXT,
                market_cap TEXT,
                pe_ratio REAL,
                currency TEXT
            )
        """)

        # Financial metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_metrics (
                ticker TEXT PRIMARY KEY,
                data TEXT
            )
        """)

        # Insert stock prices
        for ticker, data in STOCK_PRICES.items():
            cursor.execute(
                "INSERT OR REPLACE INTO stock_prices VALUES (?, ?, ?, ?, ?, ?)",
                (ticker, data["price"], data["change"], data["market_cap"],
                 data["pe_ratio"], data["currency"]),
            )

        # Insert financial metrics as JSON
        import json
        for ticker, data in FINANCIAL_METRICS.items():
            cursor.execute(
                "INSERT OR REPLACE INTO financial_metrics VALUES (?, ?)",
                (ticker, json.dumps(data)),
            )

        self.db_conn.commit()
        logger.info(f"SQLite DB built: {len(STOCK_PRICES)} stocks, "
                     f"{len(FINANCIAL_METRICS)} metrics sets")

    # ----- Query Helpers -----

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string for FAISS search."""
        return self.embedding_model.encode(
            [query], normalize_embeddings=True,
        ).astype(np.float32)

    def get_document_by_index(self, idx: int) -> dict:
        """Get document by its index position."""
        return self.documents[idx]


# ----- Module-level access -----

def get_datastore() -> DataStore:
    """Get or create the singleton DataStore."""
    return DataStore()


if __name__ == "__main__":
    """Run directly to pre-build all indexes: python -m data.ingest"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    store = get_datastore()
    print(f"\n✅ Ingestion complete!")
    print(f"   Documents: {len(store.documents)}")
    print(f"   FAISS vectors: {store.faiss_index.ntotal}")
    print(f"   Graph nodes: {store.graph.number_of_nodes()}")
    print(f"   Graph edges: {store.graph.number_of_edges()}")
