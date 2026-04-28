"""
Pydantic models for all data types — structured output at every boundary.
Every LLM input/output, API request/response, and data transfer uses these models.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --- Document Models ---

class DocumentMetadata(BaseModel):
    """Metadata attached to each document for filtering before retrieval."""
    company: str
    ticker: str
    date: str
    doc_type: str  # earnings_transcript, sec_filing, compliance_guide


class Document(BaseModel):
    """A single document in the retrieval corpus."""
    id: str
    text: str
    embedding: list[float] | None = None
    metadata: DocumentMetadata


# --- Structured Data Models ---

class StockPrice(BaseModel):
    """Deterministic stock price data — no LLM needed for this."""
    ticker: str
    price: float
    change: str
    market_cap: str
    pe_ratio: float
    currency: str


class FinancialMetrics(BaseModel):
    """Key financial metrics per stock."""
    ticker: str
    metrics: dict[str, str]


# --- Knowledge Graph Models ---

class GraphEdge(BaseModel):
    """A single edge in the knowledge graph."""
    source: str
    target: str
    relation: str  # COMPETES_WITH, SUPPLIES_TO, IS_CEO_OF, etc.


class GraphNode(BaseModel):
    """A node in the knowledge graph (company, person, etc.)."""
    id: str
    type: str  # company, person
    name: str
    sector: str | None = None
    country: str | None = None
    role: str | None = None  # for person nodes


class GraphTraversalResult(BaseModel):
    """Result of a graph traversal (BFS)."""
    ticker: str
    name: str
    sector: str
    hop: int
    path: str


class GraphEnrichment(BaseModel):
    """Graph context injected alongside retrieved docs for richer answers."""
    company: str | None = None
    sector: str | None = None
    competitors: list[str] = Field(default_factory=list)
    key_relationships: list[str] = Field(default_factory=list)
    supply_chain_exposure: list[str] = Field(default_factory=list)


# --- Query & Response Models ---

class QueryIntent(str, Enum):
    """Intent classification — routes query to the correct pipeline."""
    FACTUAL = "factual"
    COMPARISON = "comparison"
    PRICE_LOOKUP = "price_lookup"
    COMPLIANCE = "compliance"
    INVESTMENT_ADVICE = "advice"
    SUPPLY_CHAIN = "supply_chain"
    GENERAL = "general"
    BLOCKED = "blocked"


class QueryRequest(BaseModel):
    """API input model for /query endpoint."""
    query: str = Field(..., min_length=1, max_length=1000)
    jurisdiction: str | None = Field(None, description="User jurisdiction: UAE, UK, US")
    user_id: str = Field(default="demo_user")


class RetrievedDocument(BaseModel):
    """A document returned from the retrieval pipeline with score."""
    document: dict
    score: float
    text: str
    source: str


class QueryResponse(BaseModel):
    """Structured response — Pydantic validated at every LLM boundary."""
    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    intent: str
    pipeline_used: str = ""
    disclaimer: str = ""
    trace_id: str = ""
    docs_retrieved: int = 0
    graph_context: GraphEnrichment | None = None
    compliance_safe: bool = True
    numbers_verified: bool = False

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        return round(v, 3)


class LLMResponse(BaseModel):
    """Raw response parsed from the LLM — before guardrails."""
    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
