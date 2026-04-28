"""
Query Orchestrator — LangGraph StateGraph with conditional routing.
Replaces the if/else pipeline with a proper stateful graph.

Flow:
  User Query
    → classify_intent (guardrail layer 1)
    → route_by_intent (conditional edge)
    → retrieve / graph_traverse / structured_lookup / decompose / block
    → generate_response (LLM or mock)
    → validate_output (guardrail layer 3)
    → inject_disclaimer
    → return traced response (guardrail layer 4)
"""

import logging
import re
import time
from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from config.prompts import FINANCIAL_RESEARCH_PROMPT
from data.ingest import DataStore, get_datastore
from data.schema import QueryIntent, QueryResponse, GraphEnrichment
from engines.retrieval import hybrid_retrieve
from engines.knowledge_graph import (
    enrich_context, get_supply_chain_impact, traverse,
)
from engines.structured_lookup import get_price, format_price_response
from engines.llm_engine import generate as llm_generate
from guardrails.input_guard import classify_intent, get_intent_action, detect_prompt_injection
from guardrails.output_guard import scan_compliance, verify_numbers, inject_disclaimer
from guardrails.prompts_guard import build_system_prompt
from tracing.tracer import Trace

logger = logging.getLogger(__name__)


# ============================================================
# STATE DEFINITION
# ============================================================

class QueryState(TypedDict):
    """State that flows through the LangGraph pipeline."""
    query: str
    intent: str
    ticker: str | None
    retrieved_docs: list[dict]
    graph_context: dict
    nri_context: dict | None
    system_prompt: str
    response: str
    compliance_safe: bool
    numbers_verified: bool
    trace_id: str
    pipeline_used: str
    user_jurisdiction: str | None
    user_id: str
    trace: dict
    _trace_obj: Any  # internal Trace object, not serialized


# ============================================================
# TICKER EXTRACTION
# ============================================================

TICKER_MAP: dict[str, str] = {
    "NVIDIA": "NVDA", "TSMC": "TSM", "HDFC": "HDFCBANK",
    "APPLE": "AAPL", "GOOGLE": "GOOGL", "AMAZON": "AMZN",
}

KNOWN_TICKERS = [
    "NVDA", "AMD", "RELIANCE", "TCS", "INFY", "TSM", "HDFCBANK",
    "AAPL", "GOOGL", "AMZN",
]


def extract_ticker(query: str) -> str | None:
    """Extract ticker symbol from query text."""
    # Check for known tickers and company names
    all_patterns = KNOWN_TICKERS + list(TICKER_MAP.keys())
    pattern = r"\b(" + "|".join(all_patterns) + r")\b"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        raw = match.group(1).upper()
        return TICKER_MAP.get(raw, raw)
    return None


# ============================================================
# LANGGRAPH NODES
# ============================================================

def classify_node(state: QueryState) -> dict:
    """Node 1: Classify intent + extract ticker."""
    trace_obj = state.get("_trace_obj") or Trace(state["query"], state.get("user_id", "demo_user"))

    intent = classify_intent(state["query"])
    action = get_intent_action(intent)
    ticker = extract_ticker(state["query"])

    trace_obj.add_span("intent_classification", {
        "intent": intent.value,
        "action": action["action"],
        "pipeline": action["pipeline"],
        "ticker": ticker,
    })

    return {
        "intent": intent.value,
        "ticker": ticker,
        "pipeline_used": action["pipeline"] or "none",
        "_trace_obj": trace_obj,
        "trace_id": trace_obj.trace_id,
    }


def block_node(state: QueryState) -> dict:
    """Node: Block dangerous/injected queries."""
    trace_obj = state.get("_trace_obj")
    action = get_intent_action(QueryIntent.BLOCKED)

    if trace_obj:
        trace_obj.add_span("query_blocked", {"reason": "blocked_intent"})

    return {
        "response": action["message"],
        "compliance_safe": True,
        "trace": trace_obj.summary() if trace_obj else {},
    }


def structured_lookup_node(state: QueryState) -> dict:
    """Node: Direct price lookup — skip LLM entirely."""
    trace_obj = state.get("_trace_obj")
    store = get_datastore()
    ticker = state.get("ticker", "")

    price = get_price(ticker, store)
    if price:
        response = format_price_response(price)
    else:
        response = f"Price data not available for {ticker}."

    if trace_obj:
        trace_obj.add_span("structured_lookup", {
            "ticker": ticker,
            "found": price is not None,
        })

    return {
        "response": response,
        "compliance_safe": True,
        "numbers_verified": True,  # deterministic data, always verified
    }


def retrieve_node(state: QueryState) -> dict:
    """Node: Hybrid retrieval (FAISS + BM25 + RRF + reranker)."""
    trace_obj = state.get("_trace_obj")
    store = get_datastore()

    docs = hybrid_retrieve(
        state["query"], store, ticker=state.get("ticker"), top_k=3,
    )

    # Graph enrichment if we have a ticker
    graph_ctx = {}
    if state.get("ticker"):
        enrichment = enrich_context(state["ticker"], store)
        graph_ctx = enrichment.model_dump()

    # NRI context injection for jurisdiction-aware queries
    nri_ctx = None
    if state.get("user_jurisdiction"):
        nri_ctx = store.nri_contexts.get(state["user_jurisdiction"].upper())

    if trace_obj:
        trace_obj.add_span("hybrid_retrieval", {
            "docs_retrieved": len(docs),
            "ticker_filter": state.get("ticker"),
            "has_graph_context": bool(graph_ctx.get("company")),
            "has_nri_context": nri_ctx is not None,
        })

    return {
        "retrieved_docs": docs,
        "graph_context": graph_ctx,
        "nri_context": nri_ctx,
    }


def graph_traverse_node(state: QueryState) -> dict:
    """Node: Knowledge graph traversal for supply chain queries."""
    trace_obj = state.get("_trace_obj")
    store = get_datastore()
    ticker = state.get("ticker", "")

    impact = get_supply_chain_impact(ticker, max_hops=2, store=store)

    if impact:
        lines = [f"**Supply chain impact if {ticker} faces disruption:**\n"]
        for company in impact:
            lines.append(f"- **{company.name}** ({company.ticker}) — Hop {company.hop}")
        response = "\n".join(lines)
    else:
        response = f"No supply chain data found for {ticker}."

    if trace_obj:
        trace_obj.add_span("graph_traversal", {
            "ticker": ticker,
            "companies_affected": len(impact),
        })

    return {
        "response": response,
        "compliance_safe": True,
        "graph_context": enrich_context(ticker, store).model_dump() if ticker else {},
    }


def decompose_node(state: QueryState) -> dict:
    """Node: Decompose comparison into sub-queries, retrieve for each."""
    trace_obj = state.get("_trace_obj")
    store = get_datastore()

    sub_queries = _decompose_comparison(state["query"])
    all_docs = []
    for sq in sub_queries:
        docs = hybrid_retrieve(sq, store, top_k=2)
        all_docs.extend(docs)

    if trace_obj:
        trace_obj.add_span("decompose_and_retrieve", {
            "sub_queries": sub_queries,
            "total_docs": len(all_docs),
        })

    return {"retrieved_docs": all_docs}


def generate_node(state: QueryState) -> dict:
    """Node: Generate response using LLM or mock fallback."""
    trace_obj = state.get("_trace_obj")
    intent = QueryIntent(state["intent"])

    # If response already set (structured lookup, graph, blocked), skip LLM
    if state.get("response"):
        return {}

    # Build context from retrieved docs
    docs = state.get("retrieved_docs", [])
    context_parts = []
    for i, doc in enumerate(docs):
        source = doc.get("source", "Unknown")
        text = doc.get("text", "")
        context_parts.append(f"[Document {i+1} — {source}]\n{text}")

    context = "\n\n".join(context_parts) if context_parts else ""

    # Graph context string
    graph_ctx = state.get("graph_context", {})
    graph_str = ""
    if graph_ctx.get("company"):
        parts = [f"Company: {graph_ctx['company']} ({graph_ctx.get('sector', 'N/A')})"]
        if graph_ctx.get("competitors"):
            parts.append(f"Competitors: {', '.join(graph_ctx['competitors'])}")
        if graph_ctx.get("supply_chain_exposure"):
            parts.append(f"Supply chain: {', '.join(graph_ctx['supply_chain_exposure'])}")
        graph_str = "\n".join(parts)

    # Build system prompt
    system_prompt = build_system_prompt(intent, state.get("nri_context"))

    # Generate using LLM engine
    llm_response = llm_generate(
        user_message=state["query"],
        system_prompt=system_prompt,
        context=context,
        graph_context=graph_str,
        jurisdiction=state.get("user_jurisdiction", "N/A"),
        temperature=0.0 if intent in (QueryIntent.FACTUAL, QueryIntent.COMPLIANCE) else 0.3,
        trace_id=state.get("trace_id"),
    )

    response_text = llm_response.answer

    # Add redirect message for investment advice
    action = get_intent_action(intent)
    if action["action"] == "redirect" and action.get("message"):
        response_text = f"⚠️ {action['message']}\n\n{response_text}"

    if trace_obj:
        trace_obj.add_span("response_generation", {
            "response_length": len(response_text),
            "confidence": llm_response.confidence,
            "citations": llm_response.citations,
        })

    return {
        "response": response_text,
    }


def validate_node(state: QueryState) -> dict:
    """Node: Output validation — compliance scan + number verification."""
    trace_obj = state.get("_trace_obj")
    response = state.get("response", "")
    intent = QueryIntent(state["intent"])

    # Compliance scan
    compliance = scan_compliance(response)
    if not compliance["is_safe"]:
        response = (
            "I apologize, but I cannot provide this response as it may contain "
            "language that could be interpreted as investment advice. "
            "Please rephrase your question, and I'll provide factual data instead."
        )
        if trace_obj:
            trace_obj.add_span("compliance_block", {
                "violations": compliance["violations"],
            })

    # Number verification
    source_docs = [d.get("document", d) for d in state.get("retrieved_docs", [])]
    number_check = verify_numbers(response, source_docs)

    # Inject disclaimer
    response = inject_disclaimer(response, intent)

    if trace_obj:
        trace_obj.add_span("output_validation", {
            "compliance_safe": compliance["is_safe"],
            "numbers_verified": number_check["all_verified"],
        })
        trace_obj.add_span("response_delivered", {
            "final_length": len(response),
        })

    return {
        "response": response,
        "compliance_safe": compliance["is_safe"],
        "numbers_verified": number_check["all_verified"],
        "trace": trace_obj.summary() if trace_obj else {},
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_by_intent(state: QueryState) -> str:
    """Route to the correct pipeline node based on classified intent."""
    intent = state.get("intent", "factual")
    pipeline = state.get("pipeline_used", "rag")

    if intent == "blocked":
        return "block"
    if intent == "price_lookup":
        return "structured_lookup"
    if intent == "supply_chain":
        return "graph_traverse"
    if intent == "comparison":
        return "decompose"
    # Default: rag, rag_with_disclaimer, rag_with_jurisdiction, general
    return "retrieve"


# ============================================================
# HELPER: QUERY DECOMPOSITION
# ============================================================

def _decompose_comparison(query: str) -> list[str]:
    """Break a comparison query into sub-queries."""
    patterns = [
        r"compare\s+(\w+)\s+(?:vs|versus|and|with)\s+(\w+)",
        r"(\w+)\s+vs\s+(\w+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            a, b = match.group(1), match.group(2)
            topic = query.lower()
            for word in [a.lower(), b.lower(), "compare", "vs", "versus", "and", "with"]:
                topic = topic.replace(word, "")
            topic = topic.strip() or "financial performance"
            return [f"{a} {topic}", f"{b} {topic}"]

    return [query]


# ============================================================
# BUILD THE LANGGRAPH WORKFLOW
# ============================================================

def _build_workflow() -> StateGraph:
    """Build the LangGraph StateGraph with conditional routing."""
    workflow = StateGraph(QueryState)

    # Add nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("block", block_node)
    workflow.add_node("structured_lookup", structured_lookup_node)
    workflow.add_node("graph_traverse", graph_traverse_node)
    workflow.add_node("decompose", decompose_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)

    # Entry point
    workflow.set_entry_point("classify")

    # Conditional routing from classify
    workflow.add_conditional_edges("classify", route_by_intent, {
        "block": "block",
        "structured_lookup": "structured_lookup",
        "graph_traverse": "graph_traverse",
        "decompose": "decompose",
        "retrieve": "retrieve",
    })

    # After retrieval nodes → generate → validate → END
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("decompose", "generate")
    workflow.add_edge("generate", "validate")

    # Terminal nodes → validate → END
    workflow.add_edge("structured_lookup", "validate")
    workflow.add_edge("graph_traverse", "validate")
    workflow.add_edge("block", END)
    workflow.add_edge("validate", END)

    return workflow


# Compile the workflow once
_compiled_workflow = None


def _get_workflow():
    """Lazy-load the compiled workflow."""
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = _build_workflow().compile()
    return _compiled_workflow


# ============================================================
# PUBLIC API
# ============================================================

def process_query(
    query: str,
    user_jurisdiction: Optional[str] = None,
    user_id: str = "demo_user",
) -> dict:
    """
    Full query processing pipeline — the Head Waiter.
    Runs the LangGraph StateGraph and returns structured response.
    """
    trace = Trace(query, user_id)

    initial_state: QueryState = {
        "query": query,
        "intent": "",
        "ticker": None,
        "retrieved_docs": [],
        "graph_context": {},
        "nri_context": None,
        "system_prompt": "",
        "response": "",
        "compliance_safe": True,
        "numbers_verified": False,
        "trace_id": trace.trace_id,
        "pipeline_used": "",
        "user_jurisdiction": user_jurisdiction,
        "user_id": user_id,
        "trace": {},
        "_trace_obj": trace,
    }

    workflow = _get_workflow()
    result = workflow.invoke(initial_state)

    return {
        "response": result.get("response", ""),
        "intent": result.get("intent", ""),
        "pipeline_used": result.get("pipeline_used", ""),
        "docs_retrieved": len(result.get("retrieved_docs", [])),
        "graph_context": result.get("graph_context"),
        "compliance_safe": result.get("compliance_safe", True),
        "numbers_verified": result.get("numbers_verified", False),
        "trace": result.get("trace", trace.summary()),
    }
