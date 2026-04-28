"""
LLM Guardrails — the four-checkpoint security system from your interview answer.

Layer 1: INPUT guardrails (classify intent, block dangerous queries)
Layer 2: PROCESSING guardrails (system prompt rules, temperature control)
Layer 3: OUTPUT guardrails (number verification, compliance scan, disclaimer injection)
Layer 4: MONITORING (trace logging — Langfuse in production)

Key principle: guardrails should be PROGRAMMATIC, not just prompt-based.
Prompts can be jailbroken. Regex can't. Pydantic validation can't.
"""

import re
from enum import Enum
from pydantic import BaseModel, field_validator
from typing import Optional


# ============================================================
# LAYER 1: INPUT GUARDRAILS
# ============================================================

class QueryIntent(str, Enum):
    """Intent classification — like your Copilot's keyword routing."""
    FACTUAL = "factual"             # "What was NVDA revenue?" → safe, use RAG
    COMPARISON = "comparison"        # "Compare NVDA vs AMD" → safe, decompose
    PRICE_LOOKUP = "price_lookup"    # "What's NVDA price?" → skip LLM, use API
    COMPLIANCE = "compliance"        # "Can I invest as UAE NRI?" → add jurisdiction context
    INVESTMENT_ADVICE = "advice"     # "Should I buy NVDA?" → BLOCK or redirect
    SUPPLY_CHAIN = "supply_chain"    # "Who supplies to NVDA?" → use knowledge graph
    GENERAL = "general"             # "What is a PE ratio?" → general knowledge
    BLOCKED = "blocked"             # Prompt injection, harmful content → refuse


# Keywords that signal each intent
INTENT_PATTERNS = {
    QueryIntent.INVESTMENT_ADVICE: [
        r'\bshould\s+i\s+(buy|sell|invest|hold)\b',
        r'\brecommend\b.*\b(stock|invest|buy)\b',
        r'\bbest\s+stock\s+to\s+buy\b',
        r'\bguaranteed\s+return\b',
        r'\bwill\s+.+\s+(go\s+up|crash|rise|fall)\b',
    ],
    QueryIntent.PRICE_LOOKUP: [
        r'\b(current|today|latest|live)\s*(stock\s*)?price\b',
        r'\bhow\s+much\s+is\b',
        r'\bstock\s+price\b',
    ],
    QueryIntent.COMPLIANCE: [
        r'\bnri\b', r'\bdtaa\b', r'\bfema\b',
        r'\bnre\b.*\baccount\b', r'\bnro\b.*\baccount\b',
        r'\btax\b.*\b(uae|uk|us|dubai)\b',
        r'\bcan\s+i\s+invest\b.*\b(as|from)\b',
        r'\brepatriat\b',
    ],
    QueryIntent.COMPARISON: [
        r'\bcompare\b', r'\bvs\b', r'\bversus\b',
        r'\bdifference\s+between\b',
    ],
    QueryIntent.SUPPLY_CHAIN: [
        r'\bsuppl(y|ies|ier)\b', r'\bsupply\s*chain\b',
        r'\baffect(ed|s)?\b.*\b(if|when)\b',
        r'\bimpact\b.*\b(if|when)\b',
        r'\bwho\s+(makes|supplies|provides)\b',
    ],
    QueryIntent.BLOCKED: [
        r'\bignore\s+(your|all|previous)\s+(instructions|rules|prompt)\b',
        r'\bpretend\s+you\s+are\b',
        r'\bsystem\s+prompt\b',
        r'\bjailbreak\b',
        r'\bhow\s+to\s+(evade|avoid)\s+tax\b',
    ],
}


def classify_intent(query: str) -> QueryIntent:
    """
    Classify user query intent — first checkpoint before processing.
    In production: use an LLM classifier (like your Copilot's generate_prompt).
    Here: regex patterns for demonstration.
    """
    query_lower = query.lower().strip()

    # Check patterns in priority order (blocked first)
    for intent in [QueryIntent.BLOCKED, QueryIntent.INVESTMENT_ADVICE,
                   QueryIntent.PRICE_LOOKUP, QueryIntent.COMPLIANCE,
                   QueryIntent.COMPARISON, QueryIntent.SUPPLY_CHAIN]:
        for pattern in INTENT_PATTERNS.get(intent, []):
            if re.search(pattern, query_lower):
                return intent

    return QueryIntent.FACTUAL  # default: safe factual query


def get_intent_action(intent: QueryIntent) -> dict:
    """What to do for each intent type."""
    actions = {
        QueryIntent.FACTUAL: {
            "action": "proceed",
            "pipeline": "rag",
            "message": None,
        },
        QueryIntent.COMPARISON: {
            "action": "proceed",
            "pipeline": "decompose_and_rag",
            "message": None,
        },
        QueryIntent.PRICE_LOOKUP: {
            "action": "proceed",
            "pipeline": "structured_api",
            "message": "Using live data feed — no LLM needed for price lookups.",
        },
        QueryIntent.COMPLIANCE: {
            "action": "proceed",
            "pipeline": "rag_with_jurisdiction",
            "message": None,
        },
        QueryIntent.SUPPLY_CHAIN: {
            "action": "proceed",
            "pipeline": "knowledge_graph",
            "message": None,
        },
        QueryIntent.GENERAL: {
            "action": "proceed",
            "pipeline": "general_knowledge",
            "message": None,
        },
        QueryIntent.INVESTMENT_ADVICE: {
            "action": "redirect",
            "pipeline": "rag_with_disclaimer",
            "message": (
                "I can share research data, financial metrics, and analysis to help "
                "you make informed decisions — but I cannot recommend buying or selling "
                "any specific security. Let me show you the relevant data instead."
            ),
        },
        QueryIntent.BLOCKED: {
            "action": "block",
            "pipeline": None,
            "message": (
                "I'm unable to process this request. If you have a question about "
                "stocks, financial data, or investing, I'm happy to help with that."
            ),
        },
    }
    return actions.get(intent, actions[QueryIntent.FACTUAL])


# ============================================================
# LAYER 2: PROCESSING GUARDRAILS — system prompt rules
# ============================================================

SYSTEM_PROMPT_FINANCIAL = """You are a financial research assistant for Aspora's global equities platform.

HARD RULES (never violate):
1. NEVER recommend buying, selling, or holding any specific security.
2. NEVER predict future stock prices or market movements.
3. ALWAYS cite the source document for any factual claim.
4. If you cannot verify a number from the provided context, say "I could not verify this figure."
5. For tax and compliance questions, ALWAYS add: "Please consult a qualified financial advisor for personalized advice."
6. NEVER generate content that could be interpreted as personalized investment advice.
7. Present data objectively — let the user draw their own conclusions.
8. If the retrieved context doesn't contain enough information, say so honestly.

RESPONSE FORMAT:
- Lead with the direct answer to the user's question
- Support with specific data points from the retrieved documents
- Include source citations in [brackets]
- Add relevant disclaimers at the end
"""

def build_system_prompt(intent: QueryIntent, nri_context: dict | None = None) -> str:
    """Build the system prompt with intent-specific rules and NRI context."""
    prompt = SYSTEM_PROMPT_FINANCIAL

    if intent == QueryIntent.COMPLIANCE and nri_context:
        prompt += f"\n\nUSER JURISDICTION CONTEXT:\n"
        for key, value in nri_context.items():
            prompt += f"- {key}: {value}\n"

    if intent == QueryIntent.INVESTMENT_ADVICE:
        prompt += "\n\nADDITIONAL RULE: The user asked for investment advice. "
        prompt += "Present factual data ONLY. Do NOT say 'buy', 'sell', 'invest in', or 'recommend'. "
        prompt += "End with: 'This information is for research purposes only.'"

    return prompt


# ============================================================
# LAYER 3: OUTPUT GUARDRAILS
# ============================================================

class ResponseValidation(BaseModel):
    """Pydantic model for validating AI responses — structured output."""
    answer: str
    citations: list[str]
    confidence: float  # 0.0 to 1.0
    disclaimer_needed: bool = True
    compliance_safe: bool = True
    numbers_verified: bool = False

    @field_validator('confidence')
    @classmethod
    def confidence_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0 and 1')
        return v


# Phrases that should NEVER appear in financial AI output
COMPLIANCE_BLOCKLIST = [
    r'\byou\s+should\s+(buy|sell|invest|hold)\b',
    r'\bi\s+recommend\b',
    r'\bguaranteed\s+(return|profit|gain)\b',
    r'\bwill\s+definitely\s+(rise|fall|go\s+up|crash)\b',
    r'\bbuy\s+now\b',
    r'\bsell\s+immediately\b',
    r'\bcan\'t\s+go\s+wrong\b',
    r'\brisk[\s-]*free\b',
    r'\bsure\s+thing\b',
    r'\bno[\s-]*brainer\b',
]


def scan_compliance(text: str) -> dict:
    """
    Scan AI output for compliance violations.
    This is PROGRAMMATIC — regex, not LLM. Can't be jailbroken.
    """
    violations = []
    for pattern in COMPLIANCE_BLOCKLIST:
        matches = re.findall(pattern, text.lower())
        if matches:
            violations.append({
                "pattern": pattern,
                "match": matches[0] if isinstance(matches[0], str) else " ".join(matches[0]),
                "severity": "HIGH",
            })

    return {
        "is_safe": len(violations) == 0,
        "violations": violations,
        "violation_count": len(violations),
    }


def verify_numbers(response_text: str, source_documents: list[dict]) -> dict:
    """
    Extract numbers from AI response and cross-check against source documents.
    If AI says "revenue was $14.5 billion" but source says "$14.3 billion" → flag.

    This is the financial-grade hallucination check.
    """
    # Extract numbers with context from response
    number_pattern = r'[\$₹€]?\s*[\d,]+\.?\d*\s*(billion|million|crore|lakh|%|B|M|Cr|L)?'
    response_numbers = re.findall(number_pattern, response_text, re.IGNORECASE)

    # Extract numbers from source documents
    source_text = " ".join([d.get("text", "") for d in source_documents])
    source_numbers = re.findall(number_pattern, source_text, re.IGNORECASE)

    # Simple check: are the response numbers present in sources?
    # In production: more sophisticated extraction and matching
    verified = []
    unverified = []

    for num in response_numbers:
        num_str = num if isinstance(num, str) else str(num)
        if num_str and any(num_str.lower() in s.lower() for s in [source_text]):
            verified.append(num_str)
        else:
            unverified.append(num_str)

    return {
        "all_verified": len(unverified) == 0,
        "verified_count": len(verified),
        "unverified_count": len(unverified),
        "unverified_numbers": unverified[:5],  # first 5 for brevity
    }


DISCLAIMERS = {
    "general": "This information is for research and educational purposes only and does not constitute investment advice.",
    "compliance": "Tax rules vary by jurisdiction and individual circumstances. Please consult a qualified financial advisor or chartered accountant for personalized advice.",
    "comparison": "Past performance does not guarantee future results. This comparison is based on reported financial data and does not constitute a recommendation.",
    "price": "Prices shown may be delayed. Please verify with your broker for real-time quotes.",
}


def inject_disclaimer(text: str, intent: QueryIntent) -> str:
    """
    Automatically append the right disclaimer based on query intent.
    This happens PROGRAMMATICALLY — the LLM doesn't control it.
    """
    disclaimer_map = {
        QueryIntent.FACTUAL: "general",
        QueryIntent.COMPARISON: "comparison",
        QueryIntent.COMPLIANCE: "compliance",
        QueryIntent.PRICE_LOOKUP: "price",
        QueryIntent.INVESTMENT_ADVICE: "general",
        QueryIntent.GENERAL: "general",
        QueryIntent.SUPPLY_CHAIN: "general",
    }

    disclaimer_key = disclaimer_map.get(intent, "general")
    disclaimer = DISCLAIMERS[disclaimer_key]

    return f"{text}\n\n⚠️ Disclaimer: {disclaimer}"
