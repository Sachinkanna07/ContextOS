# Final local benchmark snapshot

Command: `python -m contextos.benchmarks.final` (default: 100 and 1,000 records; two iterations of ten fixed queries). Environment: Windows 11 `10.0.26200`, Python 3.13.5, deterministic local embedding, SQLite, `cl100k_base` context counter. The temporary synthetic corpora include duplicate, stale, scoped preference, project/tool, negated deployment, and long-memory records. Ground-truth relevant IDs were specified before retrieval. BM25 and dense index counts were asserted equal to each corpus size. This is one local measured run, not production or model-answer evidence.

| Memories | Strategy | Recall@5 | MRR | NDCG@10 | Candidate tokens | Compiled tokens | Mean ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | Full history | 0.300 | 0.248 | 0.231 | 34,040 | 34,040 | not measured |
| 100 | Vector | 0.800 | 0.687 | 0.693 | 3,102 | 3,102 | 20.369 |
| 100 | Hybrid | 0.950 | 0.783 | 0.826 | 2,854 | 2,854 | 20.852 |
| 100 | Hybrid + graph | 0.600 | 0.447 | 0.539 | 2,956 | 2,956 | 72.693 |
| 100 | ContextOS final | 0.900 | 0.800 | 0.781 | 2,854 | 868 | 24.419 |
| 1,000 | Full history | 0.300 | 0.248 | 0.231 | 343,640 | 343,640 | not measured |
| 1,000 | Vector | 0.600 | 0.646 | 0.638 | 3,098 | 3,098 | 193.776 |
| 1,000 | Hybrid | 0.900 | 0.783 | 0.797 | 2,884 | 2,884 | 196.635 |
| 1,000 | Hybrid + graph | 0.500 | 0.348 | 0.407 | 3,022 | 3,022 | 435.317 |
| 1,000 | ContextOS final | 0.850 | 0.800 | 0.757 | 2,884 | 650 | 207.143 |

ContextOS's weighted reduction of supplied context bodies was 69.59% at 100 and 77.46% at 1,000; compiled-body totals were 868 and 650 tokens. This is not provider-billed-token reduction. Over twenty query executions, the compiler emitted 82/58 facts and selected memories; ID-based required-source coverage was 85.71%/78.57%, stale-source facts were zero, and provenance coverage was 0 because direct fixture rows have no provenance event IDs. This does not describe normal ingestion. Context budget utilization was 21.7%/16.25% under a 200-token limit. The latest 1,000-record graph had 228 nodes/235 edges; SQLite size was 1,069,056 bytes and whole-process RSS was 122,015,744 bytes. These are synthetic evidence proxies, not answer correctness.

All twenty full-history query iterations included stale records, whereas the current-scope strategies did not. The separate seventeen-case temporal fixture yielded 1.0 ContextOS relation-classification/current-state accuracy versus 0.412/0.471 for two naive baselines; this is fixture-specific, not a real-world accuracy estimate.

Supported: ContextOS reduced measured supplied context tokens on this corpus; hybrid Recall@5 exceeded vector. Mixed: final compilation lost 0.05 Recall@5 versus raw hybrid at both sizes but had slightly higher MRR (0.800 versus 0.783). Opt-in graph augmentation degraded ranking and added latency here, so no graph-default change is justified. Answer quality was not measured. See [methodology](benchmarking.md) and the [condensed JSON snapshot](final-benchmark.json) before generalizing.
