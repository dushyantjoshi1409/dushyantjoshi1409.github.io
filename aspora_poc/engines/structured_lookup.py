"""
Structured Lookup Engine — SQLite for stock prices and financial metrics.
Direct query functions — NO LLM involved.

This is the "when NOT to use RAG" path:
  - "What's NVDA price?" → deterministic DB lookup, skip the LLM entirely
  - Faster, cheaper, more accurate than probabilistic LLM generation
"""

import json
import logging
from typing import Optional

from data.ingest import DataStore, get_datastore
from data.schema import StockPrice, FinancialMetrics

logger = logging.getLogger(__name__)


def get_price(ticker: str, store: Optional[DataStore] = None) -> StockPrice | None:
    """
    Direct price lookup — no LLM call.
    Returns structured StockPrice or None if not found.
    """
    if store is None:
        store = get_datastore()

    cursor = store.db_conn.cursor()
    cursor.execute(
        "SELECT ticker, price, change, market_cap, pe_ratio, currency "
        "FROM stock_prices WHERE ticker = ?",
        (ticker.upper(),),
    )
    row = cursor.fetchone()
    if row:
        logger.info(f"Price lookup: {ticker} → {row['currency']} {row['price']}")
        return StockPrice(
            ticker=row["ticker"],
            price=row["price"],
            change=row["change"],
            market_cap=row["market_cap"],
            pe_ratio=row["pe_ratio"],
            currency=row["currency"],
        )
    logger.warning(f"Price not found for ticker: {ticker}")
    return None


def get_metrics(ticker: str, store: Optional[DataStore] = None) -> FinancialMetrics | None:
    """
    Direct financial metrics lookup — no LLM call.
    Returns structured FinancialMetrics or None if not found.
    """
    if store is None:
        store = get_datastore()

    cursor = store.db_conn.cursor()
    cursor.execute(
        "SELECT ticker, data FROM financial_metrics WHERE ticker = ?",
        (ticker.upper(),),
    )
    row = cursor.fetchone()
    if row:
        metrics_data = json.loads(row["data"])
        logger.info(f"Metrics lookup: {ticker} → {len(metrics_data)} metrics")
        return FinancialMetrics(ticker=row["ticker"], metrics=metrics_data)
    logger.warning(f"Metrics not found for ticker: {ticker}")
    return None


def get_all_tickers(store: Optional[DataStore] = None) -> list[str]:
    """Get all available ticker symbols."""
    if store is None:
        store = get_datastore()

    cursor = store.db_conn.cursor()
    cursor.execute("SELECT ticker FROM stock_prices ORDER BY ticker")
    return [row["ticker"] for row in cursor.fetchall()]


def format_price_response(price: StockPrice) -> str:
    """Format a price lookup into a display string."""
    return (
        f"**{price.ticker} Current Price**: {price.currency} {price.price} "
        f"({price.change})\n"
        f"Market Cap: {price.market_cap} | P/E Ratio: {price.pe_ratio}"
    )
