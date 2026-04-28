"""
Query Orchestrator — the "Head Waiter" (LangGraph in production)

This is Floor 4 of your architecture diagram.
It receives a user query, classifies intent, routes to the right engine,
applies guardrails, and returns a structured response.

In production: LangGraph StateGraph with nodes and conditional edges
Here: explicit Python pipeline (like most of your Leaps flows)

Flow:
  User Query
    → Intent Classification (guardrail layer 1)
    → Route to correct engine
    → Retrieve context (hybrid search / graph / structured API)
    → Build prompt with guardrails (layer 2)
    → Generate response (simulated LLM)
    → Validate output (layer 3)
    → Inject disclaimer
    → Return traced response (layer 4)
"""

import time
from typing import Optional

from data.sample_data import STOCK_PRICES, FINANCIAL_METRICS, NRI_CONTEXTS
from engines.retrieval import hybrid_retrieve, INDEX
from engines.knowledge_graph import (
    enrich_query_with_graph, get_supply_chain_impact, get_competitors
)
from guardrails.llm_guardrails import (
    classify_intent, get_intent_action, build_system_prompt,
    scan_compliance, verify_numbers, inject_disclaimer,
    QueryIntent
)


# ============================================================
# TRACE — simulates Langfuse tracing
# In production: Langfuse v3 with contextvars propagation
# ============================================================
class Trace:
    """Lightweight trace object — like your Langfuse trace at Leaps."""

    def __init__(self, query: str, user_id: str = "demo_user"):
        self.trace_id = f"trace_{int(time.time() * 1000)}"
        self.query = query
        self.user_id = user_id
        self.spans = []
        self.start_time = time.time()
        self.metadata = {}

    def add_span(self, name: str, data: dict):
        span = {
            "span_id": f"span_{len(self.spans)}",
            "name": name,
            "timestamp": time.time(),
            "duration_ms": 0,
            "data": data,
        }
        self.spans.append(span)
        return span

    def summary(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "total_duration_ms": round((time.time() - self.start_time) * 1000, 1),
            "span_count": len(self.spans),
            "spans": [{"name": s["name"], "data_keys": list(s["data"].keys())} for s in self.spans],
        }


# ============================================================
# QUERY DECOMPOSITION — for comparison queries
# "Compare NVIDIA vs AMD" → two sub-queries
# ============================================================
def decompose_comparison(query: str) -> list[str]:
    """
    Break a comparison query into sub-queries.
    In production: LLM does this. Here: simple pattern matching.
    """
    import re

    # Try to extract company names
    patterns = [
        r'compare\s+(\w+)\s+(?:vs|versus|and|with)\s+(\w+)',
        r'(\w+)\s+vs\s+(\w+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            company_a, company_b = match.group(1), match.group(2)
            topic = query.lower().replace(company_a.lower(), "").replace(company_b.lower(), "")
            topic = re.sub(r'\b(compare|vs|versus|and|with)\b', '', topic).strip()
            if not topic:
                topic = "financial performance"
            return [
                f"{company_a} {topic}",
                f"{company_b} {topic}",
            ]

    return [query]  # can't decompose, return original


# ============================================================
# RESPONSE BUILDER — simulates LLM generation
# In production: GPT-4o / Claude API call with system prompt
# ============================================================
def build_response(query: str, context: dict, intent: QueryIntent) -> str:
    """
    Simulate LLM response generation.
    In production: this calls the LLM API with the system prompt + retrieved context.
    Here: we build a structured response from the retrieved data.
    """
    if intent == QueryIntent.PRICE_LOOKUP:
        ticker = context.get("ticker", "").upper()
        price_data = STOCK_PRICES.get(ticker)
        if price_data:
            return (
                f"**{ticker} Current Price**: {price_data['currency']} {price_data['price']} "
                f"({price_data['change']})\n"
                f"Market Cap: {price_data['market_cap']} | P/E Ratio: {price_data['pe_ratio']}"
            )
        return f"Price data not available for {ticker}."

    if intent == QueryIntent.SUPPLY_CHAIN:
        ticker = context.get("ticker", "").upper()
        impact = get_supply_chain_impact(ticker, max_hops=2)
        if impact:
            lines = [f"**Supply chain impact if {ticker} faces disruption:**\n"]
            for company in impact:
                lines.append(f"- **{company['name']}** ({company['ticker']}) — Hop {company['hop']}")
            return "\n".join(lines)
        return f"No supply chain data found for {ticker}."

    # For RAG-based queries, build response from retrieved documents
    retrieved = context.get("retrieved_docs", [])
    graph_context = context.get("graph_context", {})

    if not retrieved:
        return "I don't have enough information to answer this question based on available data."

    # Build a response from retrieved docs (simulating LLM synthesis)
    response_parts = []

    if intent == QueryIntent.COMPARISON:
        response_parts.append("**Comparison based on available data:**\n")
    elif intent == QueryIntent.COMPLIANCE:
        response_parts.append("**Compliance information:**\n")

    for i, doc in enumerate(retrieved):
        source = doc["source"]
        text = doc["text"]
        # Add citation
        response_parts.append(f"{text} [Source: {source}]")

    if graph_context and graph_context.get("competitors"):
        response_parts.append(
            f"\n**Related companies (competitors):** {', '.join(graph_context['competitors'])}"
        )

    if context.get("nri_context"):
        nri = context["nri_context"]
        response_parts.append(f"\n**For your jurisdiction:**")
        for key, value in nri.items():
            response_parts.append(f"- {value}")

    return "\n\n".join(response_parts)


# ============================================================
# MAIN ORCHESTRATOR — the full pipeline
# ============================================================
def process_query(
    query: str,
    user_jurisdiction: Optional[str] = None,
    user_id: str = "demo_user",
) -> dict:
    """
    Full query processing pipeline — the Head Waiter.

    Steps:
    1. Create trace (Langfuse)
    2. Classify intent (guardrail layer 1)
    3. Check if blocked/redirected
    4. Route to correct engine
    5. Retrieve context
    6. Build system prompt (guardrail layer 2)
    7. Generate response
    8. Validate output (guardrail layer 3)
    9. Inject disclaimer
    10. Log trace (guardrail layer 4)
    """

    # Step 1: Start trace
    trace = Trace(query, user_id)

    # Step 2: Intent classification
    intent = classify_intent(query)
    action = get_intent_action(intent)
    trace.add_span("intent_classification", {
        "intent": intent.value,
        "action": action["action"],
        "pipeline": action["pipeline"],
    })

    # Step 3: Handle blocked/redirected queries
    if action["action"] == "block":
        return {
            "response": action["message"],
            "intent": intent.value,
            "guardrail_triggered": "BLOCKED",
            "trace": trace.summary(),
        }

    # Step 4 & 5: Route and retrieve
    context = {"ticker": None, "retrieved_docs": [], "graph_context": {}, "nri_context": None}

    # Extract ticker from query (simple extraction)
    import re
    ticker_match = re.search(
        r'\b(NVDA|AMD|RELIANCE|TCS|INFY|TSM|HDFCBANK|NVIDIA|TSMC|HDFC)\b',
        query, re.IGNORECASE
    )
    ticker_map = {
        "NVIDIA": "NVDA", "TSMC": "TSM", "HDFC": "HDFCBANK",
    }
    if ticker_match:
        raw_ticker = ticker_match.group(1).upper()
        context["ticker"] = ticker_map.get(raw_ticker, raw_ticker)

    if action["pipeline"] == "structured_api":
        # Price lookup — skip LLM entirely (deterministic)
        trace.add_span("structured_api_lookup", {"ticker": context["ticker"]})

    elif action["pipeline"] == "knowledge_graph":
        # Graph traversal
        if context["ticker"]:
            context["graph_context"] = enrich_query_with_graph(context["ticker"])
            trace.add_span("graph_traversal", {"ticker": context["ticker"]})

    elif action["pipeline"] in ("rag", "rag_with_disclaimer", "rag_with_jurisdiction"):
        # Hybrid retrieval
        context["retrieved_docs"] = hybrid_retrieve(
            query, INDEX, ticker=context["ticker"], top_k=3
        )
        trace.add_span("hybrid_retrieval", {
            "docs_retrieved": len(context["retrieved_docs"]),
            "ticker_filter": context["ticker"],
        })

        # Graph enrichment (add relationship context)
        if context["ticker"]:
            context["graph_context"] = enrich_query_with_graph(context["ticker"])
            trace.add_span("graph_enrichment", {
                "competitors": context["graph_context"].get("competitors", []),
            })

        # NRI context injection
        if user_jurisdiction and action["pipeline"] == "rag_with_jurisdiction":
            context["nri_context"] = NRI_CONTEXTS.get(user_jurisdiction.upper())
            trace.add_span("nri_context_injection", {
                "jurisdiction": user_jurisdiction,
            })

    elif action["pipeline"] == "decompose_and_rag":
        # Decompose comparison → retrieve for each sub-query → combine
        sub_queries = decompose_comparison(query)
        all_docs = []
        for sq in sub_queries:
            docs = hybrid_retrieve(sq, INDEX, top_k=2)
            all_docs.extend(docs)
        context["retrieved_docs"] = all_docs
        trace.add_span("decompose_and_retrieve", {
            "sub_queries": sub_queries,
            "total_docs": len(all_docs),
        })

    # Step 6: Build system prompt
    system_prompt = build_system_prompt(intent, context.get("nri_context"))
    trace.add_span("prompt_construction", {
        "intent": intent.value,
        "has_nri_context": context.get("nri_context") is not None,
        "prompt_length": len(system_prompt),
    })

    # Step 7: Generate response
    response_text = build_response(query, context, intent)

    # Add redirect message for investment advice queries
    if action["action"] == "redirect" and action["message"]:
        response_text = f"⚠️ {action['message']}\n\n{response_text}"

    trace.add_span("response_generation", {
        "response_length": len(response_text),
        "simulated": True,  # in production: actual LLM call with token counts
    })

    # Step 8: Output validation (guardrail layer 3)
    compliance_result = scan_compliance(response_text)
    trace.add_span("compliance_scan", compliance_result)

    if not compliance_result["is_safe"]:
        response_text = (
            "I apologize, but I cannot provide this response as it may contain "
            "language that could be interpreted as investment advice. "
            "Please rephrase your question, and I'll provide factual data instead."
        )
        trace.add_span("compliance_block", {
            "violations": compliance_result["violations"],
        })

    # Number verification
    source_docs = [d.get("document", d) for d in context.get("retrieved_docs", [])]
    number_check = verify_numbers(response_text, source_docs)
    trace.add_span("number_verification", number_check)

    # Step 9: Inject disclaimer
    response_text = inject_disclaimer(response_text, intent)

    # Step 10: Final trace
    trace.add_span("response_delivered", {
        "final_length": len(response_text),
        "intent": intent.value,
        "guardrails_triggered": not compliance_result["is_safe"],
    })

    return {
        "response": response_text,
        "intent": intent.value,
        "pipeline_used": action["pipeline"],
        "docs_retrieved": len(context.get("retrieved_docs", [])),
        "graph_context": context.get("graph_context"),
        "compliance_safe": compliance_result["is_safe"],
        "numbers_verified": number_check["all_verified"],
        "trace": trace.summary(),
    }
