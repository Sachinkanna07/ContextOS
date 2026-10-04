"""ContextOS Model Service — high-level unified model execution orchestrator."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any
from uuid import uuid4

from contextos.core.enums import (
    ModelFinishReason,
    ProviderDispatchState,
    RetrievalMode,
    RoutingPolicy,
    TemporalScope,
    TokenMeasurementSource,
)
from contextos.core.models import (
    AskResult,
    CompilationConfig,
    CompiledContext,
    ContextBudget,
    ModelInvocationTelemetry,
    ModelRequest,
    ModelResponse,
    ProviderDispatchEvidence,
    RetrievalConfig,
    RetrievalQuery,
    RouteDecision,
)
from contextos.core.protocols import (
    CompilationService,
    ModelProvider,
    ModelRouter,
    RetrievalService,
    TelemetryRepository,
    TokenAwareOptimizer,
    TokenCounter,
)
from contextos.services.token_counter import get_token_counter_for_model
from contextos.storage.telemetry_repo import sanitize_telemetry_metadata

logger = logging.getLogger(__name__)


class ContextOSModelService:
    """Orchestrates ContextOS context preparation, model routing, and invocation telemetry.

    ContextOS prepares, compresses, routes, measures, and explains context.
    The model adapter is downstream.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        optimizer: TokenAwareOptimizer,
        compilation_service: CompilationService,
        router: ModelRouter,
        providers: dict[str, ModelProvider],
        telemetry_repo: TelemetryRepository,
        token_counter: TokenCounter,
        explainability_service: Any = None,
    ) -> None:
        self._retrieval = retrieval_service
        self._optimizer = optimizer
        self._compilation = compilation_service
        self._router = router
        self._providers = dict(providers)
        self._telemetry = telemetry_repo
        self._token_counter = token_counter
        self._explainability = explainability_service

    def set_explainability_service(self, explainability_service: Any) -> None:
        self._explainability = explainability_service

    def register_provider(self, provider: ModelProvider) -> None:
        """Register or replace a provider adapter."""
        self._providers[provider.provider_id] = provider

    def get_provider(self, provider_id: str) -> ModelProvider | None:
        return self._providers.get(provider_id)

    def list_providers(self) -> list[ModelProvider]:
        return list(self._providers.values())

    async def ask(
        self,
        query: str,
        system_prompt: str | None = None,
        retrieval_config: RetrievalConfig | None = None,
        compilation_config: CompilationConfig | None = None,
        routing_policy: RoutingPolicy | None = None,
        target_provider: str | None = None,
        target_model: str | None = None,
        allow_fallback: bool = False,
        allow_remote: bool = False,
        temperature: float | None = None,
        max_output_tokens: int = 1024,
        timeout_seconds: float = 30.0,
        required_capabilities: list[str] | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        explain: bool = False,
    ) -> AskResult:
        """Execute end-to-end query -> retrieval -> optimization -> compilation -> routing -> generation -> telemetry."""
        total_started = time.perf_counter()
        invocation_id = uuid4()
        req_metadata = metadata or {}

        # -------------------------------------------------------------------
        # 1. Retrieval (Hybrid + Graph + Temporal)
        # -------------------------------------------------------------------
        retrieval_result = await self._retrieval.retrieve(query, retrieval_config)
        retrieval_ms = retrieval_result.trace.total_latency_ms

        raw_candidates = retrieval_result.memories
        candidate_context_tokens = sum(
            self._token_counter.count(item.memory.content) for item in raw_candidates
        )
        lexical_candidate_count = len(retrieval_result.strategy_results.get("lexical", []))
        dense_candidate_count = len(retrieval_result.strategy_results.get("dense", []))
        graph_expanded_count = len(retrieval_result.strategy_results.get("graph", []))
        hybrid_candidate_count = len(raw_candidates)

        # Count temporal filter exclusions from retrieval stage trace if present
        temporal_filtered_count = 0
        for st in retrieval_result.trace.stages:
            if st.stage_name == "eligibility_filter":
                temporal_filtered_count = max(0, st.input_count - st.output_count)

        # -------------------------------------------------------------------
        # 2. Token-Aware Whole-Memory Optimization
        # -------------------------------------------------------------------
        comp_cfg = compilation_config or CompilationConfig()
        opt_started = time.perf_counter()
        selection = self._optimizer.optimize(
            query=query,
            candidates=raw_candidates,
            budget=ContextBudget(max_tokens=comp_cfg.budget),
        )
        optimization_ms = (time.perf_counter() - opt_started) * 1000.0
        optimized_context_tokens = selection.total_tokens
        selected_memory_count = len(selection.selected_memories)

        # -------------------------------------------------------------------
        # 3. Query-Aware Fact Context Compilation
        # -------------------------------------------------------------------
        comp_started = time.perf_counter()
        compiled_context = await self._compilation.compile(
            query=query,
            memories=selection,
            config=comp_cfg,
        )
        compilation_ms = (time.perf_counter() - comp_started) * 1000.0
        compiled_context_tokens = compiled_context.total_tokens
        compiled_fact_count = len(compiled_context.facts)

        # -------------------------------------------------------------------
        # 4. Routing Decision
        # -------------------------------------------------------------------
        model_req = ModelRequest(
            user_prompt=query,
            model=target_model,
            provider=target_provider,
            system_prompt=system_prompt,
            compiled_context=compiled_context,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            routing_policy=routing_policy,
            allow_fallback=allow_fallback,
            allow_remote=allow_remote,
            required_capabilities=required_capabilities or [],
            metadata=req_metadata,
        )

        route_decision = await self._router.route(
            request=model_req,
            providers=self._providers,
            policy=routing_policy,
        )
        routing_ms = route_decision.routing_latency_ms
        chosen_provider = self._providers[route_decision.selected_provider]

        # -------------------------------------------------------------------
        # 5. Target-Model Token Counting
        # -------------------------------------------------------------------
        tc_started = time.perf_counter()
        target_counter = get_token_counter_for_model(route_decision.selected_model)
        prompt_tokens_before = target_counter.count(query)
        if system_prompt:
            prompt_tokens_before += target_counter.count(system_prompt)

        # Target-model specific recount: ensures candidate, optimized, compiled,
        # avoided, and reduction ratio all share the exact same tokenization basis.
        target_candidate_tokens = sum(
            target_counter.count(item.memory.content) for item in raw_candidates
        )
        target_optimized_tokens = sum(
            target_counter.count(mem.memory.content) for mem in selection.selected_memories
        )
        target_compiled_tokens = (
            target_counter.count(compiled_context.context_text)
            if compiled_context.context_text
            else 0
        )
        final_input_tokens = prompt_tokens_before + target_compiled_tokens
        preflight_input_tokens = final_input_tokens
        token_counting_ms = (time.perf_counter() - tc_started) * 1000.0

        # Update model request with chosen model and invoke
        model_req.model = route_decision.selected_model
        model_req.provider = route_decision.selected_provider

        # -------------------------------------------------------------------
        # 5b. Request Construction & Fingerprinting (Provider Dispatch Receipt)
        # -------------------------------------------------------------------
        context_str = compiled_context.context_text or ""
        compiled_context_sha256 = hashlib.sha256(context_str.encode("utf-8")).hexdigest()
        # This is the actual ModelRequest handed to the provider adapter, not its
        # provider-specific HTTP payload (which each adapter constructs later).
        actual_req_context = (
            model_req.compiled_context.context_text
            if model_req.compiled_context and model_req.compiled_context.context_text
            else ""
        )
        context_match = bool(
            (context_str == actual_req_context)
            and (context_str == "" or context_str in actual_req_context)
        )
        compiled_context_in_request = context_match
        request_payload_representation = (
            f"system:{model_req.system_prompt or ''}\n"
            f"context:{actual_req_context}\n"
            f"prompt:{model_req.user_prompt}"
        )
        logical_request_sha256 = hashlib.sha256(
            request_payload_representation.encode("utf-8")
        ).hexdigest()

        dispatch_evidence = ProviderDispatchEvidence(
            provider_id=chosen_provider.provider_id,
            model_id=route_decision.selected_model,
            state=ProviderDispatchState.REQUEST_CONSTRUCTED,
            compiled_context_sha256=compiled_context_sha256,
            logical_request_sha256=logical_request_sha256,
            compiled_context_in_request=compiled_context_in_request,
            preflight_input_tokens=preflight_input_tokens,
            provider_input_tokens=None,
            provider_response_received=False,
            measurement_source=None,
            context_match=context_match,
        )

        # -------------------------------------------------------------------
        # 6. Downstream Provider Generation
        # -------------------------------------------------------------------
        dispatch_evidence.state = ProviderDispatchState.DISPATCH_ATTEMPTED
        gen_exc: Exception | None = None
        try:
            response = await chosen_provider.generate(model_req)
            if response.finish_reason == ModelFinishReason.ERROR or response.error:
                dispatch_evidence.state = ProviderDispatchState.DISPATCH_FAILED
                dispatch_evidence.provider_response_received = False
            else:
                dispatch_evidence.state = ProviderDispatchState.RESPONSE_RECEIVED
                dispatch_evidence.provider_response_received = True
                dispatch_evidence.provider_input_tokens = response.input_tokens
                dispatch_evidence.measurement_source = (
                    response.token_measurement_source.value if response.token_measurement_source else None
                )
        except Exception as exc:
            gen_exc = exc
            dispatch_evidence.state = ProviderDispatchState.DISPATCH_FAILED
            dispatch_evidence.provider_response_received = False
            # Synthesize safe failure response for failure telemetry persistence
            response = ModelResponse(
                text="",
                model_id=route_decision.selected_model,
                provider_id=chosen_provider.provider_id,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=(time.perf_counter() - total_started) * 1000.0,
                finish_reason=ModelFinishReason.ERROR,
                error=exc.__class__.__name__,
            )

        provider_latency_ms = response.latency_ms
        end_to_end_ms = (time.perf_counter() - total_started) * 1000.0

        # -------------------------------------------------------------------
        # 7. Savings and Reduction Calculation (Model-Specific Basis)
        # -------------------------------------------------------------------
        tokens_avoided = max(0, target_candidate_tokens - target_compiled_tokens)
        reduction_ratio = (
            max(0.0, 1.0 - (target_compiled_tokens / target_candidate_tokens))
            if target_candidate_tokens > 0
            else 0.0
        )

        # -------------------------------------------------------------------
        # 8. Telemetry Recording
        # -------------------------------------------------------------------
        safe_meta = sanitize_telemetry_metadata({
            "request_id": response.request_id,
            "strategy": compiled_context.strategy.value,
            **(req_metadata or {}),
        })

        telemetry = ModelInvocationTelemetry(
            invocation_id=invocation_id,
            session_id=session_id,
            provider_id=chosen_provider.provider_id,
            model_id=route_decision.selected_model,
            is_local=chosen_provider.is_local,
            candidate_context_tokens=target_candidate_tokens,
            retrieved_context_tokens=target_candidate_tokens,
            optimized_context_tokens=target_optimized_tokens,
            compiled_context_tokens=target_compiled_tokens,
            prompt_tokens_before_context=prompt_tokens_before,
            preflight_input_tokens=preflight_input_tokens,
            final_input_tokens=final_input_tokens,
            provider_input_tokens=response.input_tokens,
            provider_output_tokens=response.output_tokens,
            provider_total_tokens=response.total_tokens or (response.input_tokens + response.output_tokens),
            token_measurement_source=response.token_measurement_source,
            context_token_measurement_source=target_counter.measurement_source,
            context_tokenizer=target_counter.encoding_name,
            context_tokens_avoided=tokens_avoided,
            reduction_ratio=reduction_ratio,
            lexical_candidate_count=lexical_candidate_count,
            dense_candidate_count=dense_candidate_count,
            hybrid_candidate_count=hybrid_candidate_count,
            graph_expanded_count=graph_expanded_count,
            temporal_filtered_count=temporal_filtered_count,
            selected_memory_count=selected_memory_count,
            compiled_fact_count=compiled_fact_count,
            retrieval_ms=retrieval_ms,
            optimization_ms=optimization_ms,
            compilation_ms=compilation_ms,
            routing_ms=routing_ms,
            token_counting_ms=token_counting_ms,
            provider_latency_ms=provider_latency_ms,
            end_to_end_ms=end_to_end_ms,
            routing_policy=route_decision.policy,
            routing_reason=route_decision.reason,
            selected_provider=route_decision.selected_provider,
            selected_model=route_decision.selected_model,
            fallback_used=route_decision.fallback_used,
            fallback_reason=route_decision.fallback_reason,
            finish_reason=response.finish_reason,
            status="success" if gen_exc is None and not response.error else "error",
            error_code=response.error or (gen_exc.__class__.__name__ if gen_exc else None),
            metadata=safe_meta,
        )

        try:
            await self._telemetry.record(telemetry)
        except Exception as tel_exc:
            logger.error("Failed to persist model invocation telemetry: %s", tel_exc)

        explanation_data: dict[str, Any] | None = None
        if explain and self._explainability is not None:
            try:
                from contextos.services.explainability import ExplanationRequest
                exp_req = ExplanationRequest(
                    query=query,
                    budget=comp_cfg.budget,
                    limit=retrieval_config.max_results if retrieval_config else 25,
                    temporal_scope=(
                        TemporalScope(retrieval_config.temporal_scope.value)
                        if retrieval_config and hasattr(retrieval_config, "temporal_scope")
                        else TemporalScope.CURRENT
                    ),
                    graph=True,
                )
                trace = await self._explainability.build_trace(
                    request=exp_req,
                    retrieval_request=RetrievalQuery(
                        text=query,
                        mode=RetrievalMode.HYBRID_GRAPH,
                        k=retrieval_config.max_results if retrieval_config else 25,
                    ),
                    retrieved=retrieval_result,
                    selection=selection,
                    compiled=compiled_context,
                    dispatch_evidence=dispatch_evidence,
                    started_at=total_started,
                    pipeline_ms=retrieval_ms + optimization_ms + compilation_ms,
                )
                explanation_data = trace.model_dump(mode="json")
            except Exception as e:
                logger.warning("Failed to build explanation in ask(): %s", e)

        if gen_exc is not None:
            # Attach observable dispatch evidence, explanation, and telemetry before re-raising
            setattr(gen_exc, "dispatch_evidence", dispatch_evidence)
            setattr(gen_exc, "explanation", explanation_data)
            setattr(gen_exc, "telemetry", telemetry)
            raise gen_exc

        return AskResult(
            response=response,
            compiled_context=compiled_context,
            route_decision=route_decision,
            telemetry=telemetry,
            dispatch_evidence=dispatch_evidence,
            explanation=explanation_data,
        )
