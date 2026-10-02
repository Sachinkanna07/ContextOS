"""Ground truth metric checks independent of the live retrieval implementation."""

import asyncio

import pytest

from contextos.benchmarks.final import _measure_size, distribution, ranking_metrics


def test_ranking_metrics_reward_relevant_order_without_claiming_perfect_recall():
    good = ranking_metrics(["a", "b", "c"], {"a", "c"})
    late = ranking_metrics(["b", "a", "c"], {"a", "c"})
    assert good["recall@1"] == 0.5
    assert good["recall@3"] == 1
    assert good["mrr"] == 1
    assert late["mrr"] == 0.5
    assert good["ndcg@10"] > late["ndcg@10"]
    assert good["precision@5"] == 0.4
    with pytest.raises(ValueError):
        ranking_metrics(["a"], set())


def test_latency_distribution_reports_sample_count_and_observed_p95():
    result = distribution([1, 2, 3, 4, 5])
    assert result == {"mean_ms": 3, "median_ms": 3, "p95_ms": 5, "samples": 5}


def test_final_corpus_is_actually_indexed_and_context_evidence_is_measured():
    result = asyncio.run(_measure_size(100, 1))
    assert result["lexical_index_count"] == 100
    assert result["dense_index_count"] == 100
    contextos = result["strategies"]["contextos"]
    assert contextos["optimized_tokens_total"] <= contextos["candidate_tokens_total"]
    assert contextos["context_evidence"]["emitted"] > 0
    assert 0 <= contextos["context_evidence"]["required_source_coverage"] <= 1
    assert contextos["context_evidence"]["stale_source_facts"] == 0
    assert contextos["stage_latency"]["compiler"]["samples"] == 10
    assert result["process_rss_bytes"] > 0
