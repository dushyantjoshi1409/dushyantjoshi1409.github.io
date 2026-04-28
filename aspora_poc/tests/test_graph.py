"""Tests for knowledge graph — traversal, enrichment, path finding."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from data.ingest import get_datastore
from engines.knowledge_graph import (
    get_node, get_relationships, traverse, enrich_context,
    find_path, get_competitors, get_supply_chain_impact,
)


@pytest.fixture(scope="module")
def store():
    return get_datastore()


class TestNodeLookup:
    def test_get_existing_node(self, store):
        node = get_node("NVDA", store)
        assert node is not None
        assert node["name"] == "NVIDIA"
        assert node["type"] == "company"

    def test_get_nonexistent_node(self, store):
        node = get_node("DOESNOTEXIST", store)
        assert node is None

    def test_case_insensitive(self, store):
        node = get_node("nvda", store)
        assert node is not None


class TestRelationships:
    def test_nvidia_relationships(self, store):
        rels = get_relationships("NVDA", store=store)
        assert len(rels) > 0
        relations = [r["relation"] for r in rels]
        assert "COMPETES_WITH" in relations or "SUPPLIES_TO" in relations

    def test_filter_by_relation(self, store):
        rels = get_relationships("NVDA", "COMPETES_WITH", store)
        assert len(rels) > 0
        assert all(r["relation"] == "COMPETES_WITH" for r in rels)


class TestGraphTraversal:
    def test_tsmc_supply_chain(self, store):
        results = traverse("TSM", "SUPPLIES_TO", max_hops=2, store=store)
        assert len(results) > 0
        tickers = [r.ticker for r in results]
        # TSMC supplies to NVDA, AMD, AAPL
        assert "NVDA" in tickers or "AMD" in tickers

    def test_hop_tracking(self, store):
        results = traverse("TSM", "SUPPLIES_TO", max_hops=3, store=store)
        # Should have hop 1 (direct) and hop 2 (indirect)
        hops = set(r.hop for r in results)
        assert 1 in hops

    def test_max_hops_limit(self, store):
        results_1 = traverse("TSM", "SUPPLIES_TO", max_hops=1, store=store)
        results_3 = traverse("TSM", "SUPPLIES_TO", max_hops=3, store=store)
        # More hops should find more or equal nodes
        assert len(results_3) >= len(results_1)


class TestEnrichContext:
    def test_nvidia_enrichment(self, store):
        enrichment = enrich_context("NVDA", store)
        assert enrichment.company == "NVIDIA"
        assert enrichment.sector == "Semiconductors"
        assert len(enrichment.competitors) > 0

    def test_nonexistent_ticker(self, store):
        enrichment = enrich_context("DOESNOTEXIST", store)
        assert enrichment.company is None


class TestFindPath:
    def test_path_tsm_to_googl(self, store):
        path = find_path("TSM", "GOOGL", store)
        assert path is not None
        assert path[0] == "TSM"
        assert path[-1] == "GOOGL"

    def test_no_path(self, store):
        # These might not have a path
        path = find_path("HDFCBANK", "AAPL", store)
        # Can be None or have a path — just check it doesn't crash


class TestConvenienceFunctions:
    def test_get_competitors(self, store):
        competitors = get_competitors("NVDA", store)
        names = [c["target_name"] for c in competitors]
        assert "AMD" in names

    def test_supply_chain_impact(self, store):
        impact = get_supply_chain_impact("TSM", max_hops=2, store=store)
        assert len(impact) > 0
