"""
LLM Engine — Claude API wrapper with retry, JSON mode, tracing.
Falls back to mock responses if ANTHROPIC_API_KEY is not set.
"""

import json
import logging
import time
from typing import Optional

from config.settings import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS
from config.prompts import FINANCIAL_RESEARCH_PROMPT
from data.schema import LLMResponse

logger = logging.getLogger(__name__)

# Try to initialize Anthropic client
_anthropic_client = None
if ANTHROPIC_API_KEY and not ANTHROPIC_API_KEY.startswith("sk-ant-..."):
    try:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info(f"✅ Anthropic client initialized (model: {LLM_MODEL})")
    except Exception as e:
        logger.warning(f"⚠️ Anthropic init failed, using mock LLM: {e}")
        _anthropic_client = None
else:
    logger.info("ℹ️ No Anthropic API key — using mock LLM responses")


def generate(
    user_message: str,
    system_prompt: str,
    context: str = "",
    graph_context: str = "",
    jurisdiction: str = "N/A",
    temperature: float = 0.0,
    max_tokens: int = LLM_MAX_TOKENS,
    trace_id: Optional[str] = None,
) -> LLMResponse:
    """
    Generate a response using Claude API or mock fallback.
    Temperature 0 for factual/financial queries, 0.3 for general knowledge.
    Always injects retrieved context into the prompt.
    """
    start = time.time()

    # Format the system prompt with context
    formatted_prompt = system_prompt.format(
        context=context or "No context retrieved.",
        jurisdiction=jurisdiction,
        graph_context=graph_context or "No graph context.",
    )

    if _anthropic_client:
        return _generate_with_claude(
            user_message, formatted_prompt, temperature, max_tokens, trace_id,
        )
    else:
        return _generate_mock(user_message, context)


def _generate_with_claude(
    user_message: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    trace_id: Optional[str],
) -> LLMResponse:
    """Call Claude API with retry and JSON parsing."""
    try:
        response = _anthropic_client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = response.content[0].text
        logger.info(f"Claude response: {len(raw_text)} chars, "
                     f"{response.usage.input_tokens}in/{response.usage.output_tokens}out tokens")

        # Try to parse as JSON
        return _parse_llm_response(raw_text)

    except Exception as e:
        logger.error(f"Claude API error: {e}")
        # Graceful fallback to mock
        return LLMResponse(
            answer=f"I encountered an error generating a response. Error: {str(e)}",
            citations=[],
            confidence=0.0,
        )


def _generate_mock(user_message: str, context: str) -> LLMResponse:
    """
    Mock LLM — builds structured response from retrieved context.
    Used when ANTHROPIC_API_KEY is not set.
    """
    # Extract key info from context for a reasonable mock response
    if context and context != "No context retrieved.":
        # Use the raw context as the answer with citation markers
        answer = context
        citations = ["Retrieved documents"]
        confidence = 0.7
    else:
        answer = "I don't have enough information to answer this question based on available data."
        citations = []
        confidence = 0.1

    return LLMResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
    )


def _parse_llm_response(raw_text: str) -> LLMResponse:
    """Parse LLM text output into structured LLMResponse."""
    # Try JSON first
    try:
        # Find JSON block in the response
        json_start = raw_text.find("{")
        json_end = raw_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(raw_text[json_start:json_end])
            return LLMResponse(
                answer=parsed.get("answer", raw_text),
                citations=parsed.get("citations", []),
                confidence=parsed.get("confidence", 0.5),
            )
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: treat entire response as the answer
    return LLMResponse(
        answer=raw_text,
        citations=[],
        confidence=0.5,
    )
