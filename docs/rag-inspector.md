# RAG Inspector

`contextos inspect "query"` and `POST /api/v1/inspect` execute one retrieval/selection/compilation explanation pipeline and return bounded structured evidence. No model provider is invoked. The API request bounds query length, result limit, budget, and optional target model. Graph augmentation is off unless `--graph` is supplied.

```powershell
contextos inspect "Which tool does Atlas use?" --mode hybrid --limit 10 --budget 300
contextos inspect "Which tool does Atlas use?" --graph --memory <uuid> --compare --json
contextos inspect "Which tool does Atlas use?" --target-model qwen --show-content
```

The result contains stage input/output/removed counts and measured latency, candidate rank/channel scores, temporal and graph evidence, optimizer decision, compiler transformations, a target-memory reason when requested, and a context-token diff. Scores are kept on their native scales; rank-fusion evidence is not presented as a percentage. Graph paths come from the explanation execution, not a second search performed for display.

`candidate_tokens` recounts retrieved memory bodies, `optimized_tokens` recounts selected bodies, and `compiled_tokens` counts the actual compiled context text. Compiler framing can make compiled tokens exceed candidate tokens; `net_token_change` therefore remains signed, while avoided tokens floor at zero. The counter source and tokenizer are returned. A requested target model recounts all three values on the selected counter, but does not rewrite the already-compiled context or prove provider-billed token savings. `provenance_coverage` is the fraction of emitted facts with retained provenance IDs, not a factuality score.

`--compare` makes four additional bounded lexical/dense/hybrid/hybrid+graph retrievals, reporting overlap, rank movement, and latency. It does not have ground truth and consequently reports no Recall, MRR, NDCG, answer quality, or winner. The separate benchmark supplies fixed relevance labels.

Query text and memory content are omitted by default. `--show-content` is a deliberate local disclosure. JSON remains bounded and follows the same content flag. Explanation provenance can be incomplete; an absence reason must be `NOT_AVAILABLE` when candidate or graph channel membership cannot be proven from retained evidence. Provider state is `NOT_ATTEMPTED`.
