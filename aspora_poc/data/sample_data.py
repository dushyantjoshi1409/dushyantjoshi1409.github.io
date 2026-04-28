"""
Sample financial data for the Level Up Stock POC.
In production, this comes from ingestion pipelines (Kafka + Airflow).
Here we simulate all three data types:
  1. Documents (for vector/semantic search) — earnings transcripts, filings
  2. Structured data (for time-series/API lookup) — stock prices, metrics
  3. Relationships (for knowledge graph) — company connections
"""

# ============================================================
# 1. DOCUMENTS — would live in Pinecone in production
#    Each doc has: id, text, metadata (company, date, type)
#    Metadata enables filtering BEFORE semantic search
# ============================================================
DOCUMENTS = [
    {
        "id": "doc_001",
        "text": "In Q3 2024, NVIDIA reported data center revenue of $14.5 billion, up 112% year-over-year. CEO Jensen Huang stated that demand for AI infrastructure continues to accelerate across every industry.",
        "metadata": {"company": "NVIDIA", "ticker": "NVDA", "date": "2024-Q3", "type": "earnings_transcript"}
    },
    {
        "id": "doc_002",
        "text": "AMD reported Q3 2024 data center revenue of $3.5 billion, up 122% year-over-year. CEO Lisa Su highlighted the MI300X GPU as a strong competitor in AI training workloads.",
        "metadata": {"company": "AMD", "ticker": "AMD", "date": "2024-Q3", "type": "earnings_transcript"}
    },
    {
        "id": "doc_003",
        "text": "NVIDIA's Q2 2024 data center revenue was $10.3 billion. The company announced partnerships with major cloud providers for next-generation AI infrastructure deployment.",
        "metadata": {"company": "NVIDIA", "ticker": "NVDA", "date": "2024-Q2", "type": "earnings_transcript"}
    },
    {
        "id": "doc_004",
        "text": "AMD's Q2 2024 data center segment generated $2.8 billion in revenue. The company shipped over 100,000 MI300X accelerators to hyperscale customers.",
        "metadata": {"company": "AMD", "ticker": "AMD", "date": "2024-Q2", "type": "earnings_transcript"}
    },
    {
        "id": "doc_005",
        "text": "Reliance Industries reported consolidated revenue of Rs 2,35,481 crore for Q2 FY2025. Jio Platforms contributed Rs 37,120 crore. The retail segment generated Rs 76,302 crore.",
        "metadata": {"company": "Reliance Industries", "ticker": "RELIANCE", "date": "2024-Q2", "type": "earnings_transcript"}
    },
    {
        "id": "doc_006",
        "text": "TCS reported revenue of Rs 64,259 crore for Q2 FY2025, a growth of 7.6% year-over-year. The company added 5 new deals worth over $100 million each. BFSI vertical grew 2.4%.",
        "metadata": {"company": "TCS", "ticker": "TCS", "date": "2024-Q2", "type": "earnings_transcript"}
    },
    {
        "id": "doc_007",
        "text": "Infosys raised its revenue growth guidance to 3.75-4.5% for FY2025. CEO Salil Parekh noted strong demand in AI and cloud transformation services. Large deal TCV was $4.1 billion.",
        "metadata": {"company": "Infosys", "ticker": "INFY", "date": "2024-Q2", "type": "earnings_transcript"}
    },
    {
        "id": "doc_008",
        "text": "TSMC reported Q3 2024 revenue of NT$759.69 billion, up 36% year-over-year. Advanced 3nm and 5nm processes accounted for 69% of wafer revenue. AI-related demand drove the growth.",
        "metadata": {"company": "TSMC", "ticker": "TSM", "date": "2024-Q3", "type": "earnings_transcript"}
    },
    {
        "id": "doc_009",
        "text": "For NRIs investing from the UAE, the India-UAE Double Taxation Avoidance Agreement (DTAA) provides significant benefits. Capital gains from Indian mutual funds are taxable only in the UAE, and since UAE has no capital gains tax, the effective tax rate is zero.",
        "metadata": {"company": "REGULATORY", "ticker": "NA", "date": "2024", "type": "compliance_guide"}
    },
    {
        "id": "doc_010",
        "text": "Under FEMA regulations, NRIs can invest in Indian equities through NRE or NRO accounts. NRE account investments are freely repatriable. NRO investments have a repatriation limit of USD 1 million per financial year after applicable taxes.",
        "metadata": {"company": "REGULATORY", "ticker": "NA", "date": "2024", "type": "compliance_guide"}
    },
    {
        "id": "doc_011",
        "text": "NVIDIA's risk factors include concentration in a few large customers, export restrictions to China, supply chain dependence on TSMC for advanced chip manufacturing, and increasing competition from AMD and custom AI chips from Google and Amazon.",
        "metadata": {"company": "NVIDIA", "ticker": "NVDA", "date": "2024", "type": "sec_filing_10k"}
    },
    {
        "id": "doc_012",
        "text": "HDFC Bank reported net profit of Rs 16,821 crore for Q2 FY2025. Net interest income was Rs 30,110 crore. The bank added 1,400 new branches during the quarter. Asset quality improved with gross NPA at 1.24%.",
        "metadata": {"company": "HDFC Bank", "ticker": "HDFCBANK", "date": "2024-Q2", "type": "earnings_transcript"}
    },
]


# ============================================================
# 2. STRUCTURED DATA — would live in TimescaleDB in production
#    Stock prices, financial metrics — deterministic lookups
#    These should NEVER go through LLM — direct API/DB query
# ============================================================
STOCK_PRICES = {
    "NVDA": {"price": 875.30, "change": "+2.4%", "market_cap": "$2.15T", "pe_ratio": 65.2, "currency": "USD"},
    "AMD": {"price": 178.50, "change": "+1.1%", "market_cap": "$288B", "pe_ratio": 48.7, "currency": "USD"},
    "RELIANCE": {"price": 2485.60, "change": "-0.3%", "market_cap": "₹16.8L Cr", "pe_ratio": 28.5, "currency": "INR"},
    "TCS": {"price": 3842.15, "change": "+0.7%", "market_cap": "₹14.1L Cr", "pe_ratio": 31.2, "currency": "INR"},
    "INFY": {"price": 1678.30, "change": "+1.2%", "market_cap": "₹6.9L Cr", "pe_ratio": 27.8, "currency": "INR"},
    "TSM": {"price": 185.20, "change": "+1.8%", "market_cap": "$958B", "pe_ratio": 29.1, "currency": "USD"},
    "HDFCBANK": {"price": 1625.40, "change": "+0.5%", "market_cap": "₹12.4L Cr", "pe_ratio": 19.8, "currency": "INR"},
}

FINANCIAL_METRICS = {
    "NVDA": {"revenue_q3": "$18.1B", "revenue_yoy": "+94%", "dc_revenue": "$14.5B", "gross_margin": "74.0%"},
    "AMD": {"revenue_q3": "$6.8B", "revenue_yoy": "+18%", "dc_revenue": "$3.5B", "gross_margin": "52.1%"},
    "RELIANCE": {"revenue_q2": "₹2,35,481 Cr", "revenue_yoy": "+4.7%", "net_profit": "₹19,101 Cr", "gross_margin": "N/A"},
    "TCS": {"revenue_q2": "₹64,259 Cr", "revenue_yoy": "+7.6%", "net_profit": "₹12,380 Cr", "gross_margin": "N/A"},
    "INFY": {"revenue_q2": "₹40,986 Cr", "revenue_yoy": "+4.2%", "net_profit": "₹6,506 Cr", "guidance": "3.75-4.5%"},
    "HDFCBANK": {"revenue_q2": "₹30,110 Cr NII", "net_profit": "₹16,821 Cr", "npa_gross": "1.24%", "branches": "8,900+"},
}


# ============================================================
# 3. KNOWLEDGE GRAPH — would live in Neo4j in production
#    Nodes = companies, people, sectors
#    Edges = relationships (COMPETES_WITH, SUPPLIES_TO, etc.)
#    Enables: "If TSMC has issues, who's affected?"
# ============================================================
KNOWLEDGE_GRAPH = {
    "nodes": {
        "NVDA": {"type": "company", "name": "NVIDIA", "sector": "Semiconductors", "country": "US"},
        "AMD": {"type": "company", "name": "AMD", "sector": "Semiconductors", "country": "US"},
        "TSM": {"type": "company", "name": "TSMC", "sector": "Semiconductors", "country": "Taiwan"},
        "RELIANCE": {"type": "company", "name": "Reliance Industries", "sector": "Conglomerate", "country": "India"},
        "TCS": {"type": "company", "name": "TCS", "sector": "IT Services", "country": "India"},
        "INFY": {"type": "company", "name": "Infosys", "sector": "IT Services", "country": "India"},
        "HDFCBANK": {"type": "company", "name": "HDFC Bank", "sector": "Banking", "country": "India"},
        "GOOGL": {"type": "company", "name": "Google/Alphabet", "sector": "Technology", "country": "US"},
        "AMZN": {"type": "company", "name": "Amazon", "sector": "Technology", "country": "US"},
        "AAPL": {"type": "company", "name": "Apple", "sector": "Technology", "country": "US"},
        "jensen_huang": {"type": "person", "name": "Jensen Huang", "role": "CEO"},
        "lisa_su": {"type": "person", "name": "Lisa Su", "role": "CEO"},
        "salil_parekh": {"type": "person", "name": "Salil Parekh", "role": "CEO"},
    },
    "edges": [
        # Competition
        {"from": "NVDA", "to": "AMD", "relation": "COMPETES_WITH"},
        {"from": "TCS", "to": "INFY", "relation": "COMPETES_WITH"},
        # Supply chain
        {"from": "TSM", "to": "NVDA", "relation": "SUPPLIES_TO"},
        {"from": "TSM", "to": "AMD", "relation": "SUPPLIES_TO"},
        {"from": "TSM", "to": "AAPL", "relation": "SUPPLIES_TO"},
        {"from": "TSM", "to": "GOOGL", "relation": "SUPPLIES_TO"},
        # Customer relationships
        {"from": "NVDA", "to": "GOOGL", "relation": "SUPPLIES_TO"},
        {"from": "NVDA", "to": "AMZN", "relation": "SUPPLIES_TO"},
        {"from": "TCS", "to": "HDFCBANK", "relation": "PROVIDES_SERVICES_TO"},
        {"from": "INFY", "to": "RELIANCE", "relation": "PROVIDES_SERVICES_TO"},
        # Executives
        {"from": "jensen_huang", "to": "NVDA", "relation": "IS_CEO_OF"},
        {"from": "lisa_su", "to": "AMD", "relation": "IS_CEO_OF"},
        {"from": "salil_parekh", "to": "INFY", "relation": "IS_CEO_OF"},
        # Sector membership
        {"from": "NVDA", "to": "AMD", "relation": "SAME_SECTOR"},
        {"from": "TCS", "to": "INFY", "relation": "SAME_SECTOR"},
    ]
}


# ============================================================
# 4. NRI CONTEXT — jurisdiction-specific rules
#    Injected into prompts based on user profile
# ============================================================
NRI_CONTEXTS = {
    "UAE": {
        "dtaa": "India-UAE DTAA: Capital gains taxable only in UAE. UAE has no capital gains tax → effective rate 0%.",
        "account_types": "Can use NRE (freely repatriable) or NRO (limit $1M/year repatriation).",
        "restrictions": "Cannot invest in agricultural land, plantation, or farmhouse in India.",
        "tax_tip": "Long-term equity gains above ₹1.25 lakh taxed at 12.5% in India, but refundable under DTAA.",
    },
    "UK": {
        "dtaa": "India-UK DTAA: Tax credit available for taxes paid in India. UK taxes worldwide income.",
        "account_types": "NRE or NRO accounts. NRE interest is tax-free in India.",
        "restrictions": "Standard NRI investment rules apply. FEMA compliance required.",
        "tax_tip": "UK residents pay CGT on Indian investments but can claim credit for Indian TDS.",
    },
    "US": {
        "dtaa": "India-US DTAA: Complex — both countries may tax. Foreign tax credit available.",
        "account_types": "NRE or NRO. FATCA reporting obligations in US.",
        "restrictions": "Cannot invest in certain Indian mutual funds (PFIC rules in US make it unfavorable).",
        "tax_tip": "US residents face PFIC penalties on Indian mutual funds. Direct equity may be preferable.",
    },
}
