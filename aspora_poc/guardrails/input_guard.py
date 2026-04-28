"""
Input Guardrails — Layer 1: Intent classification + prompt injection detection.
Programmatic (regex-based), not just prompt-based. Regex can't be jailbroken.
"""

import logging
import re
from typing import Optional

from data.schema import QueryIntent

logger = logging.getLogger(__name__)


# Keywords that signal each intent — checked in priority order
INTENT_PATTERNS: dict[QueryIntent, list[str]] = {
    QueryIntent.BLOCKED: [
        r"\bignore\s+(your|all|previous)\s+(instructions|rules|prompt)\b",
        r"\bpretend\s+you\s+are\b",
        r"\bsystem\s+prompt\b",
        r"\bjailbreak\b",
        r"\bhow\s+to\s+(evade|avoid)\s+tax\b",
    ],
    QueryIntent.INVESTMENT_ADVICE: [
        r"\bshould\s+i\s+(buy|sell|invest|hold)\b",
        r"\brecommend\b.*\b(stock|invest|buy)\b",
        r"\bbest\s+stock\s+to\s+buy\b",
        r"\bguaranteed\s+return\b",
        r"\bwill\s+.+\s+(go\s+up|crash|rise|fall)\b",
    ],
    QueryIntent.PRICE_LOOKUP: [
        r"\b(current|today|latest|live)\s*(stock\s*)?price\b",
        r"\bhow\s+much\s+is\b",
        r"\bstock\s+price\b",
    ],
    QueryIntent.COMPLIANCE: [
        r"\bnri\b", r"\bdtaa\b", r"\bfema\b",
        r"\bnre\b.*\baccount\b", r"\bnro\b.*\baccount\b",
        r"\btax\b.*\b(uae|uk|us|dubai)\b",
        r"\bcan\s+i\s+invest\b.*\b(as|from)\b",
        r"\brepatriat\b",
    ],
    QueryIntent.COMPARISON: [
        r"\bcompare\b", r"\bvs\b", r"\bversus\b",
        r"\bdifference\s+between\b",
    ],
    QueryIntent.SUPPLY_CHAIN: [
        r"\bsuppl(y|ies|ier)\b", r"\bsupply\s*chain\b",
        r"\baffect(ed|s)?\b.*\b(if|when)\b",
        r"\bimpact\b.*\b(if|when)\b",
        r"\bwho\s+(makes|supplies|provides)\b",
    ],
}


def classify_intent(query: str) -> QueryIntent:
    """
    Classify user query intent — first checkpoint before processing.
    Regex-based for reliability; in production could add an LLM classifier layer.
    """
    query_lower = query.lower().strip()

    # Check patterns in priority order (blocked first, then advice, etc.)
    for intent in [
        QueryIntent.BLOCKED,
        QueryIntent.INVESTMENT_ADVICE,
        QueryIntent.PRICE_LOOKUP,
        QueryIntent.COMPLIANCE,
        QueryIntent.COMPARISON,
        QueryIntent.SUPPLY_CHAIN,
    ]:
        for pattern in INTENT_PATTERNS.get(intent, []):
            if re.search(pattern, query_lower):
                logger.info(f"Intent classified: {intent.value} (pattern: {pattern})")
                return intent

    return QueryIntent.FACTUAL  # default: safe factual query


def detect_prompt_injection(query: str) -> bool:
    """
    Detect prompt injection attempts — programmatic check.
    Returns True if injection detected.
    """
    injection_patterns = [
        r"ignore\s+(all\s+)?(previous|above)\s+(instructions|rules|prompts?)",
        r"you\s+are\s+now\s+",
        r"pretend\s+(to\s+be|you\s+are)",
        r"forget\s+(everything|your\s+instructions)",
        r"new\s+instructions?\s*:",
        r"system\s*:\s*",
        r"<\s*/?system\s*>",
    ]

    query_lower = query.lower()
    for pattern in injection_patterns:
        if re.search(pattern, query_lower):
            logger.warning(f"⚠️ Prompt injection detected: {pattern}")
            return True
    return False


def get_intent_action(intent: QueryIntent) -> dict:
    """What to do for each intent type — routing decisions."""
    from config.prompts import INVESTMENT_ADVICE_REDIRECT, BLOCKED_RESPONSE

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
            "message": INVESTMENT_ADVICE_REDIRECT,
        },
        QueryIntent.BLOCKED: {
            "action": "block",
            "pipeline": None,
            "message": BLOCKED_RESPONSE,
        },
    }
    return actions.get(intent, actions[QueryIntent.FACTUAL])
