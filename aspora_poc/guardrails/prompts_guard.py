"""
Processing Guardrails — System prompt builder with intent-specific rules.
All prompts are referenced from config/prompts.py, never hardcoded.
"""

import logging
from typing import Optional

from config.prompts import FINANCIAL_RESEARCH_PROMPT
from data.schema import QueryIntent

logger = logging.getLogger(__name__)


def build_system_prompt(
    intent: QueryIntent,
    nri_context: Optional[dict] = None,
) -> str:
    """
    Build the system prompt with intent-specific rules and NRI context.
    Uses FINANCIAL_RESEARCH_PROMPT as the base template.
    """
    prompt = FINANCIAL_RESEARCH_PROMPT

    # Inject jurisdiction-specific context for compliance queries
    if intent == QueryIntent.COMPLIANCE and nri_context:
        extra = "\n\nUSER JURISDICTION CONTEXT:\n"
        for key, value in nri_context.items():
            extra += f"- {key}: {value}\n"
        prompt += extra

    # Add extra restrictions for investment advice queries
    if intent == QueryIntent.INVESTMENT_ADVICE:
        prompt += (
            "\n\nADDITIONAL RULE: The user asked for investment advice. "
            "Present factual data ONLY. Do NOT say 'buy', 'sell', 'invest in', or 'recommend'. "
            "End with: 'This information is for research purposes only.'"
        )

    return prompt
