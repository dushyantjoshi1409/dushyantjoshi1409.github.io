"""
Tracing — Langfuse wrapper with graceful fallback to file logging.
If Langfuse API keys are configured, traces go to Langfuse dashboard.
Otherwise, traces are logged to file for local observability.
"""

import json
import logging
import time
from typing import Any, Optional

from config.settings import (
    LANGFUSE_ENABLED, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
)

logger = logging.getLogger(__name__)

# Try to initialize Langfuse client
_langfuse_client = None
if LANGFUSE_ENABLED:
    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        logger.info("✅ Langfuse tracing enabled")
    except Exception as e:
        logger.warning(f"⚠️ Langfuse init failed, falling back to file logging: {e}")
        _langfuse_client = None


class Trace:
    """
    Lightweight trace object — wraps Langfuse or falls back to structured logging.
    Tracks: spans, timing, token counts, retrieval scores.
    """

    def __init__(self, query: str, user_id: str = "demo_user") -> None:
        self.trace_id = f"trace_{int(time.time() * 1000)}"
        self.query = query
        self.user_id = user_id
        self.spans: list[dict] = []
        self.start_time = time.time()
        self.metadata: dict[str, Any] = {}

        # Create Langfuse trace if available
        self._langfuse_trace = None
        if _langfuse_client:
            try:
                self._langfuse_trace = _langfuse_client.trace(
                    id=self.trace_id,
                    name="query_pipeline",
                    user_id=user_id,
                    input={"query": query},
                )
            except Exception as e:
                logger.warning(f"Langfuse trace creation failed: {e}")

    def add_span(self, name: str, data: dict) -> dict:
        """Add a span to the trace with timing data."""
        span = {
            "span_id": f"span_{len(self.spans)}",
            "name": name,
            "timestamp": time.time(),
            "duration_ms": round((time.time() - self.start_time) * 1000, 1),
            "data": data,
        }
        self.spans.append(span)

        # Log to Langfuse if available
        if self._langfuse_trace:
            try:
                self._langfuse_trace.span(
                    name=name,
                    input=data,
                )
            except Exception:
                pass  # graceful degradation

        # Always log to file as well
        logger.debug(f"Span [{name}]: {json.dumps(data, default=str)}")
        return span

    def set_output(self, output: dict) -> None:
        """Set the final trace output."""
        if self._langfuse_trace:
            try:
                self._langfuse_trace.update(output=output)
            except Exception:
                pass

    def summary(self) -> dict:
        """Get a summary of the trace for API responses."""
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "total_duration_ms": round((time.time() - self.start_time) * 1000, 1),
            "span_count": len(self.spans),
            "spans": [
                {"name": s["name"], "data_keys": list(s["data"].keys())}
                for s in self.spans
            ],
        }
