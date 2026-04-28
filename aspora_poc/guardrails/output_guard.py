"""
Output Guardrails — Layer 3: Compliance scan, number verification, disclaimer injection.
All programmatic (regex + Pydantic), not prompt-based.
"""

import logging
import re

from config.prompts import DISCLAIMERS
from data.schema import QueryIntent

logger = logging.getLogger(__name__)


# Phrases that should NEVER appear in financial AI output
COMPLIANCE_BLOCKLIST: list[str] = [
    r"\byou\s+should\s+(buy|sell|invest|hold)\b",
    r"\bi\s+recommend\b",
    r"\bguaranteed\s+(return|profit|gain)\b",
    r"\bwill\s+definitely\s+(rise|fall|go\s+up|crash)\b",
    r"\bbuy\s+now\b",
    r"\bsell\s+immediately\b",
    r"\bcan't\s+go\s+wrong\b",
    r"\brisk[\s-]*free\b",
    r"\bsure\s+thing\b",
    r"\bno[\s-]*brainer\b",
]


def scan_compliance(text: str) -> dict:
    """
    Scan AI output for compliance violations.
    PROGRAMMATIC — regex, not LLM. Can't be jailbroken.
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

    result = {
        "is_safe": len(violations) == 0,
        "violations": violations,
        "violation_count": len(violations),
    }

    if violations:
        logger.warning(f"⚠️ Compliance violations found: {len(violations)}")
    return result


def verify_numbers(response_text: str, source_documents: list[dict]) -> dict:
    """
    Extract numbers from AI response and cross-check against source documents.
    If AI says "revenue was $14.5 billion" but source says "$14.3 billion" → flag.
    This is the financial-grade hallucination check.
    """
    number_pattern = r"[\$₹€]?\s*[\d,]+\.?\d*\s*(billion|million|crore|lakh|%|B|M|Cr|L)?"
    response_numbers = re.findall(number_pattern, response_text, re.IGNORECASE)

    source_text = " ".join([d.get("text", "") for d in source_documents])
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
        "unverified_numbers": unverified[:5],
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
