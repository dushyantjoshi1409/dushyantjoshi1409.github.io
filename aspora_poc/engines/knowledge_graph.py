"""
Knowledge Graph Engine — NetworkX-based graph with BFS traversal.
Upgraded from dict-based graph to proper NetworkX with typed traversal.

Enables queries like:
  - "If TSMC has supply issues, who's affected?" → BFS traversal
  - "Who competes with NVIDIA?" → direct relationship lookup
  - "What companies does Jensen Huang lead?" → person→company lookup
  - "Path from TSMC to Amazon?" → shortest path
"""

import logging
from typing import Optional

import networkx as nx

from data.ingest import DataStore, get_datastore
from data.schema import GraphTraversalResult, GraphEnrichment

logger = logging.getLogger(__name__)


def _get_graph(store: Optional[DataStore] = None) -> nx.MultiDiGraph:
    """Get the knowledge graph from the DataStore."""
    if store is None:
        store = get_datastore()
    return store.graph


# -------------------------------------------------------
# 1. NODE LOOKUP
# -------------------------------------------------------
def get_node(ticker: str, store: Optional[DataStore] = None) -> dict | None:
    """Get a node's attributes from the graph by ticker/id."""
    graph = _get_graph(store)
    ticker = ticker.upper()
    if ticker in graph.nodes:
        return dict(graph.nodes[ticker])
    return None


# -------------------------------------------------------
# 2. RELATIONSHIP LOOKUP
# -------------------------------------------------------
def get_relationships(
    ticker: str,
    relation_type: Optional[str] = None,
    store: Optional[DataStore] = None,
) -> list[dict]:
    """
    Get all relationships for a given node (both directions).
    Like Neo4j: MATCH (n {ticker: 'NVDA'})-[r]-(m) RETURN r, m
    """
    graph = _get_graph(store)
    ticker = ticker.upper()
    results = []

    # Outgoing edges
    if ticker in graph.nodes:
        for _, target, _, data in graph.out_edges(ticker, data=True, keys=True):
            if relation_type is None or data.get("relation") == relation_type:
                target_attrs = dict(graph.nodes.get(target, {}))
                results.append({
                    "relation": data.get("relation", "UNKNOWN"),
                    "target": target,
                    "target_name": target_attrs.get("name", target),
                    "target_type": target_attrs.get("type", "unknown"),
                    "direction": "outgoing",
                })

        # Incoming edges
        for source, _, _, data in graph.in_edges(ticker, data=True, keys=True):
            if relation_type is None or data.get("relation") == relation_type:
                source_attrs = dict(graph.nodes.get(source, {}))
                results.append({
                    "relation": data.get("relation", "UNKNOWN"),
                    "target": source,
                    "target_name": source_attrs.get("name", source),
                    "target_type": source_attrs.get("type", "unknown"),
                    "direction": "incoming",
                })

    return results


# -------------------------------------------------------
# 3. BFS TRAVERSAL — multi-hop graph expansion
# -------------------------------------------------------
def traverse(
    start: str,
    relation: str,
    max_hops: int = 3,
    store: Optional[DataStore] = None,
) -> list[GraphTraversalResult]:
    """
    BFS traversal up to N hops following a specific relation type.
    Like Neo4j: MATCH (start)-[:SUPPLIES_TO*1..3]->(affected) RETURN affected

    Example: "If TSMC has supply issues, who's affected?"
      TSM -[SUPPLIES_TO]-> NVDA (hop 1)
      TSM -[SUPPLIES_TO]-> AMD (hop 1)
      NVDA -[SUPPLIES_TO]-> GOOGL (hop 2)
    """
    graph = _get_graph(store)
    start = start.upper()
    visited = set()
    results = []
    queue = [(start, 0, [start])]  # (node, hop, path)

    while queue:
        current, hop, path = queue.pop(0)
        if current in visited or hop > max_hops:
            continue
        visited.add(current)

        for _, target, _, data in graph.out_edges(current, data=True, keys=True):
            if data.get("relation") == relation and target not in visited:
                target_attrs = dict(graph.nodes.get(target, {}))
                new_path = path + [target]
                results.append(GraphTraversalResult(
                    ticker=target,
                    name=target_attrs.get("name", target),
                    sector=target_attrs.get("sector", "Unknown"),
                    hop=hop + 1,
                    path=" → ".join(new_path),
                ))
                queue.append((target, hop + 1, new_path))

    logger.info(f"Graph traversal: {start} -[{relation}]-> "
                f"{len(results)} nodes in ≤{max_hops} hops")
    return results


# -------------------------------------------------------
# 4. ENRICH CONTEXT — graph context for RAG injection
# -------------------------------------------------------
def enrich_context(
    ticker: str,
    store: Optional[DataStore] = None,
) -> GraphEnrichment:
    """
    Graph-enriched context for RAG.
    Before doing document retrieval, we ask the graph:
    "What else should I know about this company?"
    This context gets injected into the LLM prompt alongside retrieved documents.
    """
    node = get_node(ticker, store)
    if not node:
        return GraphEnrichment()

    competitors = get_relationships(ticker, "COMPETES_WITH", store)
    all_rels = get_relationships(ticker, store=store)
    supply_impact = traverse(ticker, "SUPPLIES_TO", max_hops=2, store=store)

    return GraphEnrichment(
        company=node.get("name"),
        sector=node.get("sector"),
        competitors=[r["target_name"] for r in competitors],
        key_relationships=[
            f"{r['target_name']} ({r['relation']})" for r in all_rels[:5]
        ],
        supply_chain_exposure=[
            f"{s.name} (hop {s.hop})" for s in supply_impact
        ],
    )


# -------------------------------------------------------
# 5. SHORTEST PATH — path between two companies
# -------------------------------------------------------
def find_path(
    source: str,
    target: str,
    store: Optional[DataStore] = None,
) -> list[str] | None:
    """
    Find shortest path between two companies in the graph.
    Returns list of node IDs in the path, or None if no path exists.
    """
    graph = _get_graph(store)
    source, target = source.upper(), target.upper()
    try:
        # Use undirected view for path finding (relationships go both ways conceptually)
        path = nx.shortest_path(graph.to_undirected(), source, target)
        logger.info(f"Path found: {' → '.join(path)}")
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        logger.info(f"No path between {source} and {target}")
        return None


# -------------------------------------------------------
# 6. CONVENIENCE FUNCTIONS
# -------------------------------------------------------
def get_competitors(ticker: str, store: Optional[DataStore] = None) -> list[dict]:
    """Get all companies that compete with the given ticker."""
    return get_relationships(ticker, "COMPETES_WITH", store)


def get_supply_chain_impact(
    ticker: str,
    max_hops: int = 2,
    store: Optional[DataStore] = None,
) -> list[GraphTraversalResult]:
    """If {ticker} has supply issues, which companies are affected?"""
    return traverse(ticker, "SUPPLIES_TO", max_hops, store)
