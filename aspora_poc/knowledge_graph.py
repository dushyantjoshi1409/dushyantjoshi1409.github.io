"""
Knowledge Graph Engine — the "Family Tree for Companies"

In production: Neo4j with Cypher queries and APOC path expansion
Here: Python dict-based graph with BFS traversal up to N hops

This enables queries like:
  - "If TSMC has supply issues, who's affected?" → graph traversal
  - "Who competes with NVIDIA?" → direct relationship lookup
  - "What companies does Jensen Huang lead?" → person-company lookup
"""

from data.sample_data import KNOWLEDGE_GRAPH


def get_node(ticker: str) -> dict | None:
    """Get a node from the graph by ticker/id."""
    return KNOWLEDGE_GRAPH["nodes"].get(ticker.upper())


def get_relationships(ticker: str, relation_type: str | None = None) -> list[dict]:
    """
    Get all relationships for a given node.
    Like Neo4j: MATCH (n {ticker: 'NVDA'})-[r]->(m) RETURN r, m
    """
    ticker = ticker.upper()
    results = []
    for edge in KNOWLEDGE_GRAPH["edges"]:
        if edge["from"] == ticker:
            if relation_type is None or edge["relation"] == relation_type:
                target_node = KNOWLEDGE_GRAPH["nodes"].get(edge["to"], {})
                results.append({
                    "relation": edge["relation"],
                    "target": edge["to"],
                    "target_name": target_node.get("name", edge["to"]),
                    "target_type": target_node.get("type", "unknown"),
                    "direction": "outgoing",
                })
        if edge["to"] == ticker:
            if relation_type is None or edge["relation"] == relation_type:
                source_node = KNOWLEDGE_GRAPH["nodes"].get(edge["from"], {})
                results.append({
                    "relation": edge["relation"],
                    "target": edge["from"],
                    "target_name": source_node.get("name", edge["from"]),
                    "target_type": source_node.get("type", "unknown"),
                    "direction": "incoming",
                })
    return results


def graph_traverse(ticker: str, relation_type: str, max_hops: int = 3) -> list[dict]:
    """
    BFS traversal up to N hops — like Neo4j APOC path expansion.

    In production (Neo4j Cypher):
      MATCH (start {ticker: 'TSM'})-[:SUPPLIES_TO*1..3]->(affected)
      RETURN affected.name, affected.ticker

    Example: "If TSMC has supply issues, who's affected?"
      TSM -[SUPPLIES_TO]-> NVDA (hop 1)
      TSM -[SUPPLIES_TO]-> AMD (hop 1)
      TSM -[SUPPLIES_TO]-> AAPL (hop 1)
      NVDA -[SUPPLIES_TO]-> GOOGL (hop 2)
      NVDA -[SUPPLIES_TO]-> AMZN (hop 2)
    """
    visited = set()
    results = []
    queue = [(ticker.upper(), 0)]  # (node, current_hop)

    while queue:
        current, hop = queue.pop(0)
        if current in visited or hop > max_hops:
            continue
        visited.add(current)

        for edge in KNOWLEDGE_GRAPH["edges"]:
            if edge["relation"] == relation_type and edge["from"] == current:
                target = edge["to"]
                if target not in visited:
                    target_node = KNOWLEDGE_GRAPH["nodes"].get(target, {})
                    results.append({
                        "ticker": target,
                        "name": target_node.get("name", target),
                        "sector": target_node.get("sector", "Unknown"),
                        "hop": hop + 1,
                        "path": f"{ticker} → {target}" if hop == 0 else f"{ticker} → ... → {current} → {target}",
                    })
                    queue.append((target, hop + 1))

    return results


def get_competitors(ticker: str) -> list[dict]:
    """Get all companies that compete with the given ticker."""
    return get_relationships(ticker, "COMPETES_WITH")


def get_supply_chain_impact(ticker: str, max_hops: int = 2) -> list[dict]:
    """
    "If {ticker} has supply issues, which companies are affected?"
    Traverses SUPPLIES_TO relationships up to max_hops.
    """
    return graph_traverse(ticker, "SUPPLIES_TO", max_hops)


def enrich_query_with_graph(ticker: str) -> dict:
    """
    Graph-enriched context for RAG.
    Before doing document retrieval, we ask the graph:
    "What else should I know about this company?"

    This context gets injected into the LLM prompt alongside retrieved documents.
    """
    node = get_node(ticker)
    if not node:
        return {"enrichment": None}

    competitors = get_competitors(ticker)
    relationships = get_relationships(ticker)
    supply_impact = get_supply_chain_impact(ticker, max_hops=2)

    return {
        "company": node.get("name"),
        "sector": node.get("sector"),
        "competitors": [r["target_name"] for r in competitors],
        "key_relationships": [
            f"{r['target_name']} ({r['relation']})" for r in relationships[:5]
        ],
        "supply_chain_exposure": [
            f"{s['name']} (hop {s['hop']})" for s in supply_impact
        ],
    }
