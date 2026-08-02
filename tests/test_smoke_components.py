"""Fast unit tests for the Milestone 0 smoke gate (Issue #14, amendment P fast 1–6).

These are NOT marked ``smoke`` — they run in the fast tier. They cover:

1. fixture tokenizer/corpus digest stability + fingerprint binding (distinct
   digest representations — review correction #5);
2. exact toy-model parameter count (from arrays) + deterministic initialization;
3. analytical-backward finite-gradient behavior + predeclared loss-improvement
   criterion;
4. quiescent snapshot construction + complete descriptor coverage;
5. cursor and counter progression (update-boundary invariants — correction #4);
6. non-mutating comparison probes + closed diagnostics.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from expertforge.checkpoints.models import CounterSnapshot
from expertforge.config.resolve import canonical_bytes, resolve_config
from expertforge.identity.fingerprint import specification_fingerprint
from expertforge.rng.manager import RngManager
from expertforge.smoke import fixtures
from expertforge.smoke.comparison import compare, extract_computational_state
from expertforge.smoke.data import (
    SmokeDataCursor,
    compute_data_config_digest_hex,
)
from expertforge.smoke.fixtures import (
    SMOKE_DATASET_INPUT_NAME,
    SMOKE_TOKENIZER_INPUT_NAME,
)
from expertforge.smoke.model import MODEL_ARCHITECTURE, SmokeModel
from expertforge.smoke.optimizer import SmokeAdamW
from expertforge.smoke.report import ComparisonReport
from expertforge.smoke.scheduler import SmokeScheduler
from expertforge.smoke.state import SmokeRuntime, SmokeStateProvider

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
GATE_CONFIG = CONFIGS / "m0-smoke-gate.yaml"

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SHA256_PREFIXED = re.compile(r"^sha256:[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Fast 1: fixture digest stability + fingerprint binding (correction #5)
# ---------------------------------------------------------------------------


class TestFixtureDigests:
    def test_corpus_is_below_1kib_and_16_symbol_alphabet(self) -> None:
        corpus = fixtures.corpus_bytes()
        assert len(corpus) < 1024
        definition = fixtures.load_tokenizer_definition()
        symbols = set(definition["symbols"])
        assert len(symbols) == 16
        # Every corpus byte is one of the 16 symbols.
        text = corpus.decode("utf-8")
        assert set(text).issubset(symbols)

    def test_distinct_digest_representations_never_conflated(self) -> None:
        # ImmutableInput / DataIdentity use the RAW hex (correction #5).
        assert HEX64.fullmatch(fixtures.corpus_digest_hex())
        assert HEX64.fullmatch(fixtures.tokenizer_digest_hex())
        # Manifest references use the PREFIXED digest.
        assert SHA256_PREFIXED.fullmatch(fixtures.corpus_content_digest())
        assert SHA256_PREFIXED.fullmatch(fixtures.tokenizer_content_digest())
        # The prefixed form is exactly "sha256:" + the raw hex.
        assert fixtures.corpus_content_digest() == f"sha256:{fixtures.corpus_digest_hex()}"
        assert fixtures.tokenizer_content_digest() == f"sha256:{fixtures.tokenizer_digest_hex()}"

    def test_corpus_digest_is_stable(self) -> None:
        # The committed corpus bytes have a fixed SHA-256 (recorded in the report).
        assert (
            fixtures.corpus_digest_hex()
            == "7a2e7c2351b5d4359db64427908d31b0bece901b8b4cf8c3bcdd11d1c2de3d6a"
        )
        assert (
            fixtures.tokenizer_digest_hex()
            == "56a8723252da3ebc78bf092467f583d6bf755d43128d37c27d8eb4940a48a338"
        )

    def test_data_config_digest_is_independent_of_corpus_digest(self) -> None:
        # clarification 5152259715 #1: data_config_digest hashes the continuation
        # contract, NOT the corpus bytes.
        digest = compute_data_config_digest_hex(
            dataset_digest_hex=fixtures.corpus_digest_hex(),
            sampler_type="sequential",
            sampler_version=1,
            batch_size=2,
            sequence_length=8,
            drop_last=True,
            tokenizer_identity="smoke-printable16-v1",
        )
        assert HEX64.fullmatch(digest)
        assert digest != fixtures.corpus_digest_hex()

    def test_specification_fingerprint_is_identical_across_independent_emits(
        self, tmp_path: Path
    ) -> None:
        env = resolve_config(GATE_CONFIG)
        cb = canonical_bytes(env)
        inputs = fixtures.fixture_immutable_inputs()
        fp1 = specification_fingerprint(cb, inputs)
        fp2 = specification_fingerprint(cb, inputs)
        assert fp1.digest_str == fp2.digest_str
        # The fingerprint embeds the corpus + tokenizer immutable inputs.
        names = {ii.name for ii in fp1.immutable_inputs}
        assert SMOKE_DATASET_INPUT_NAME in names
        assert SMOKE_TOKENIZER_INPUT_NAME in names

    def test_manifest_references_bind_to_fingerprint_inputs(self) -> None:
        env = resolve_config(GATE_CONFIG)
        fp = specification_fingerprint(canonical_bytes(env), fixtures.fixture_immutable_inputs())
        inputs = {ii.name: ii for ii in fp.immutable_inputs}
        dataset_ref = fixtures.dataset_reference()
        tokenizer_ref = fixtures.tokenizer_reference()
        assert dataset_ref.content_digest == f"sha256:{inputs[SMOKE_DATASET_INPUT_NAME].digest}"
        assert tokenizer_ref.content_digest == f"sha256:{inputs[SMOKE_TOKENIZER_INPUT_NAME].digest}"


# ---------------------------------------------------------------------------
# Fast 2: exact parameter count + deterministic initialization
# ---------------------------------------------------------------------------


class TestModelIdentity:
    def _new_manager(self) -> RngManager:
        env = resolve_config(GATE_CONFIG)
        mgr = RngManager.from_config(env.config, component="smoke")
        mgr.initialize()
        return mgr

    def test_exact_parameter_count_from_arrays(self) -> None:
        env = resolve_config(GATE_CONFIG)
        model = SmokeModel(env.config.model)
        model.initialize(self._new_manager().generator)
        # 2560 = embed(16*16) + qkv(16*48) + attn_out(16*16) + ffn(16*32) +
        # ffn_out(32*16) + lm_head(16*16)
        assert model.parameter_count() == 2560

    def test_parameter_count_is_sum_of_array_sizes(self) -> None:
        env = resolve_config(GATE_CONFIG)
        model = SmokeModel(env.config.model)
        model.initialize(self._new_manager().generator)
        total = sum(arr.size for arr in model.parameters.values())
        assert model.parameter_count() == total

    def test_deterministic_initialization_same_seed_identical_params(self) -> None:
        env = resolve_config(GATE_CONFIG)
        m1 = SmokeModel(env.config.model)
        m1.initialize(self._new_manager().generator)
        m2 = SmokeModel(env.config.model)
        m2.initialize(self._new_manager().generator)
        for name in m1.parameters:
            np.testing.assert_array_equal(m1.parameters[name], m2.parameters[name])

    def test_every_declared_dimension_controls_real_behavior(self) -> None:
        env = resolve_config(GATE_CONFIG)
        cfg = env.config.model
        model = SmokeModel(cfg)
        model.initialize(self._new_manager().generator)
        shapes = model.parameter_shapes()
        # dim controls embed/lm_head and the per-block qkv/attn_out widths;
        # ffn_dim controls the FFN inner width; n_layers controls block count;
        # n_heads controls the attention head factorization. Every declared dim
        # must appear in real shapes.
        assert shapes["embed"] == (16, cfg.dim)
        assert shapes["lm_head"] == (cfg.dim, 16)
        # n_layers blocks exist.
        block_names = [n for n in shapes if n.startswith("blocks.")]
        assert len(block_names) == cfg.n_layers * 4  # qkv/attn_out/ffn_in/ffn_out
        assert shapes["blocks.0.qkv"] == (cfg.dim, 3 * cfg.dim)
        assert shapes["blocks.0.attn_out"] == (cfg.dim, cfg.dim)
        assert shapes["blocks.0.ffn_in"] == (cfg.dim, cfg.ffn_dim)
        assert shapes["blocks.0.ffn_out"] == (cfg.ffn_dim, cfg.dim)
        # n_heads is honored by the attention factorization (head_dim = dim/n_heads).
        assert model.head_dim == cfg.dim // cfg.n_heads

    def test_model_architecture_name_is_truthful(self) -> None:
        assert MODEL_ARCHITECTURE == "smoke-toy-causal-decoder"


# ---------------------------------------------------------------------------
# Fast 3: analytical-backward finite-gradient check + loss criterion
# ---------------------------------------------------------------------------


class TestBackwardAndLossMovement:
    def _setup(self) -> tuple[SmokeModel, np.ndarray, np.ndarray]:
        env = resolve_config(GATE_CONFIG)
        mgr = RngManager.from_config(env.config, component="smoke")
        mgr.initialize()
        model = SmokeModel(env.config.model)
        model.initialize(mgr.generator)
        rng = np.random.default_rng(123)
        x = rng.integers(0, 16, size=(2, 4))
        targets = rng.integers(0, 16, size=(2, 4))
        return model, x, targets

    def test_analytical_backward_matches_finite_differences(self) -> None:
        # Verify the analytical backward against finite differences for BOTH the
        # frozen gate config (n_layers=1, n_heads=1) and a multi-head/multi-layer
        # config (proving F9: n_heads and n_layers genuinely control behavior).
        # float64 to separate real bugs from float32 noise.
        for cfg_dict in (
            {"dim": 16, "n_layers": 1, "n_heads": 1, "ffn_dim": 32},
            {"dim": 16, "n_layers": 2, "n_heads": 2, "ffn_dim": 32},
            {"dim": 24, "n_layers": 2, "n_heads": 3, "ffn_dim": 24},
        ):
            self._check_grad(cfg_dict, seed=1)

    def _check_grad(self, cfg_dict: dict[str, int], *, seed: int) -> None:
        from expertforge.config.models import ModelConfig

        cfg = ModelConfig(**cfg_dict)
        rng = np.random.default_rng(seed)
        V = 16
        scale = 0.3
        P: dict[str, np.ndarray] = {
            "embed": rng.standard_normal((V, cfg.dim)) * scale,
            "lm_head": rng.standard_normal((cfg.dim, V)) * scale,
        }
        for i in range(cfg.n_layers):
            p = f"blocks.{i}"
            P[f"{p}.qkv"] = rng.standard_normal((cfg.dim, 3 * cfg.dim)) * scale
            P[f"{p}.attn_out"] = rng.standard_normal((cfg.dim, cfg.dim)) * scale
            P[f"{p}.ffn_in"] = rng.standard_normal((cfg.dim, cfg.ffn_dim)) * scale
            P[f"{p}.ffn_out"] = rng.standard_normal((cfg.ffn_dim, cfg.dim)) * scale
        P = {k: v.astype(np.float64) for k, v in P.items()}
        model = SmokeModel(cfg)
        model.parameters = P
        model._initialized = True
        x = rng.integers(0, V, size=(2, 4))
        targets = rng.integers(0, V, size=(2, 4))
        _, cache = model.forward(x)
        grads = model.backward(cache, targets)
        eps = 1e-6

        def loss_of(params: dict[str, np.ndarray]) -> float:
            mm = SmokeModel(cfg)
            mm.parameters = params
            mm._initialized = True
            lg, _ = mm.forward(x)
            m = lg.max(axis=-1, keepdims=True)
            p = np.exp(lg - m)
            p /= p.sum(axis=-1, keepdims=True)
            B, T, _ = lg.shape
            return float(
                -np.log(p[np.arange(B)[:, None], np.arange(T)[None, :], targets] + 1e-30).mean()
            )

        for name in P:
            flat = P[name].reshape(-1)
            ana = grads[name].reshape(-1)
            for idx in range(min(8, flat.size)):
                pp = {k: v.copy() for k, v in P.items()}
                pm = {k: v.copy() for k, v in P.items()}
                pp[name].reshape(-1)[idx] += eps
                pm[name].reshape(-1)[idx] -= eps
                num = (loss_of(pp) - loss_of(pm)) / (2 * eps)
                assert abs(num - ana[idx]) < 1e-4, f"{name}[{idx}] num={num} ana={ana[idx]}"

    def test_forward_backward_produce_finite_values(self) -> None:
        model, x, targets = self._setup()
        logits, cache = model.forward(x)
        assert np.all(np.isfinite(logits))
        grads = model.backward(cache, targets)
        for name, g in grads.items():
            assert np.all(np.isfinite(g)), f"non-finite grad in {name}"

    def test_loss_improvement_criterion_is_met(self) -> None:
        env = resolve_config(GATE_CONFIG)
        cfg = env.config
        threshold = cfg.evaluation.loss_improvement_threshold
        assert threshold is not None and threshold > 0
        seq_len = cfg.training.seq_len or cfg.data.seq_len
        mgr = RngManager.from_config(cfg, component="smoke")
        mgr.initialize()
        model = SmokeModel(cfg.model)
        model.initialize(mgr.generator)
        opt = SmokeAdamW(lr=cfg.training.lr)
        opt.initialize(model.parameters)
        cursor = SmokeDataCursor.create(batch_size=cfg.training.batch_size, sequence_length=seq_len)

        def ce(logits: np.ndarray, tgt: np.ndarray) -> float:
            m = logits.max(axis=-1, keepdims=True)
            p = np.exp(logits - m)
            p /= p.sum(axis=-1, keepdims=True)
            B, T, _ = logits.shape
            return float(
                -np.log(p[np.arange(B)[:, None], np.arange(T)[None, :], tgt] + 1e-12).mean()
            )

        # initial loss before any training
        b = cursor.next_batch()
        lg, _ = model.forward(b[:, :-1])
        initial = ce(lg, b[:, 1:])
        # train a handful of updates
        for _ in range(8):
            b = cursor.next_batch()
            lg, c = model.forward(b[:, :-1])
            g = model.backward(c, b[:, 1:])
            opt.update(model.parameters, g)
        b = cursor.next_batch()
        lg, _ = model.forward(b[:, :-1])
        final = ce(lg, b[:, 1:])
        assert np.isfinite(initial) and np.isfinite(final)
        assert (initial - final) >= threshold


# ---------------------------------------------------------------------------
# Fast 4: quiescent snapshot construction + descriptor coverage
# ---------------------------------------------------------------------------


def _runtime_at_k(k: int = 2) -> SmokeRuntime:
    env = resolve_config(GATE_CONFIG)
    cfg = env.config
    seq_len = cfg.training.seq_len or cfg.data.seq_len
    mgr = RngManager.from_config(cfg, component="smoke")
    mgr.initialize()
    model = SmokeModel(cfg.model)
    model.initialize(mgr.generator)
    opt = SmokeAdamW(lr=cfg.training.lr)
    opt.initialize(model.parameters)
    sched = SmokeScheduler(lr=cfg.training.lr)
    sched.initialize()
    cursor = SmokeDataCursor.create(batch_size=cfg.training.batch_size, sequence_length=seq_len)
    counters = CounterSnapshot(
        global_update=0,
        completed_microsteps=0,
        accumulation_position=0,
        accepted_samples=0,
        accepted_sequences=0,
        processed_tokens=0,
    )
    rt = SmokeRuntime(
        model_params=model.parameters,
        model_buffers={},
        optimizer_state=opt,
        scheduler=sched,
        cursor=cursor,
        counters=counters,
        rng_manager=mgr,
    )
    m = SmokeModel(cfg.model)
    m.parameters = rt.model_params
    m._initialized = True
    tpu = seq_len * cfg.training.batch_size
    for _ in range(k):
        b = rt.cursor.next_batch()
        lg, c = m.forward(b[:, :-1])
        g = m.backward(c, b[:, 1:])
        rt.optimizer_state.update(m.parameters, g)
        rt.scheduler.step()
        nu = rt.counters.global_update + 1
        rt.counters = CounterSnapshot(
            global_update=nu,
            completed_microsteps=nu,
            accumulation_position=0,
            accepted_samples=rt.cursor.accepted_samples,
            accepted_sequences=rt.cursor.accepted_sequences,
            processed_tokens=nu * tpu,
        )
    return rt


class TestQuiescentSnapshot:
    def test_snapshot_covers_every_component(self) -> None:
        rt = _runtime_at_k(2)
        captured = SmokeStateProvider(rt).capture_checkpoint_snapshot()
        # Parameters + buffers.
        assert len(captured.parameters) == 6
        # Optimizer slots: 2 slots (exp_avg, exp_avg_sq) x 6 params = 12.
        assert len(captured.optimizer_slots) == 12
        assert captured.optimizer is None  # multi-slot path, no legacy tensor
        assert captured.optimizer_scalar_state is not None
        assert captured.scheduler is not None
        assert captured.scheduler_scalar_state is not None
        assert len(captured.rng_bundle_bytes) > 0
        # Descriptors.
        assert len(captured.model_descriptor.parameters) == 6
        assert len(captured.optimizer_descriptor.state_slots) == 12
        assert captured.optimizer_descriptor.optimizer_type == "adamw"
        assert captured.scheduler_descriptor.scheduler_type == "constant_with_step"
        assert captured.topology_descriptor.world_size == 1

    def test_snapshot_is_quiescent(self) -> None:
        rt = _runtime_at_k(2)
        captured = SmokeStateProvider(rt).capture_checkpoint_snapshot()
        assert captured.optimizer_update_complete is True
        assert captured.accumulation_position == 0
        assert captured.async_prefetch_active is False
        assert captured.counters.accumulation_position == 0

    def test_optimizer_slots_bind_to_descriptor(self) -> None:
        rt = _runtime_at_k(2)
        captured = SmokeStateProvider(rt).capture_checkpoint_snapshot()
        slot_names = {t.logical_name for t in captured.optimizer_slots}
        declared = {
            f"optimizer.{slot.slot_name}.{slot.param_name}"
            for slot in captured.optimizer_descriptor.state_slots
        }
        assert slot_names == declared


# ---------------------------------------------------------------------------
# Fast 5: cursor and counter progression (correction #4 invariants)
# ---------------------------------------------------------------------------


class TestCursorCounters:
    def test_update_boundary_invariants(self) -> None:
        rt = _runtime_at_k(2)
        c = rt.counters
        seq_len = 8
        batch_size = 2
        tpu = batch_size * seq_len
        # global_update == completed_microsteps
        assert c.global_update == c.completed_microsteps == 2
        # accumulation_position == 0
        assert c.accumulation_position == 0
        # processed_tokens == global_update * tokens_per_update
        assert c.processed_tokens == c.global_update * tpu
        # accepted_samples == global_update * batch_size
        assert c.accepted_samples == c.global_update * batch_size
        assert c.accepted_sequences == c.accepted_samples

    def test_cursor_position_progresses_consistently(self) -> None:
        cursor = SmokeDataCursor.create(batch_size=2, sequence_length=8)
        length = cursor.length
        _ = cursor.next_batch()
        # 1 batch = 2 samples * 8 tokens = 16 tokens consumed.
        assert cursor.accepted_samples == 2
        assert cursor.position == (2 * 8) % length

    def test_counters_never_regress(self) -> None:
        rt = _runtime_at_k(1)
        prev_processed = rt.counters.processed_tokens
        prev_accepted = rt.counters.accepted_samples
        # advance one more update on the existing runtime.
        seq_len = 8
        batch_size = 2
        tpu = seq_len * batch_size
        model = SmokeModel(resolve_config(GATE_CONFIG).config.model)
        model.parameters = rt.model_params
        model._initialized = True
        b = rt.cursor.next_batch()
        lg, c = model.forward(b[:, :-1])
        g = model.backward(c, b[:, 1:])
        rt.optimizer_state.update(model.parameters, g)
        rt.scheduler.step()
        nu = rt.counters.global_update + 1
        rt.counters = CounterSnapshot(
            global_update=nu,
            completed_microsteps=nu,
            accumulation_position=0,
            accepted_samples=rt.cursor.accepted_samples,
            accepted_sequences=rt.cursor.accepted_sequences,
            processed_tokens=nu * tpu,
        )
        assert rt.counters.processed_tokens > prev_processed
        assert rt.counters.accepted_samples > prev_accepted


# ---------------------------------------------------------------------------
# Fast 6: non-mutating comparison probes + closed diagnostics
# ---------------------------------------------------------------------------


class TestComparisonProbes:
    def test_probes_do_not_mutate_canonical_state(self) -> None:
        rt = _runtime_at_k(2)
        rng_before = rt.rng_manager.capture_state().to_deterministic_json()
        pos_before = rt.cursor.position
        accepted_before = rt.cursor.accepted_samples
        _ = extract_computational_state(
            model_params=rt.model_params,
            model_buffers={},
            optimizer=rt.optimizer_state,
            scheduler=rt.scheduler,
            cursor=rt.cursor,
            counters=rt.counters,
            rng_manager=rt.rng_manager,
            validation_loss=2.5,
            generated_sample=np.array([1, 2, 3, 4]),
        )
        rng_after = rt.rng_manager.capture_state().to_deterministic_json()
        assert rng_before == rng_after
        assert rt.cursor.position == pos_before
        assert rt.cursor.accepted_samples == accepted_before

    def test_equal_states_compare_equal(self) -> None:
        rt = _runtime_at_k(2)
        s1 = extract_computational_state(
            model_params=rt.model_params,
            model_buffers={},
            optimizer=rt.optimizer_state,
            scheduler=rt.scheduler,
            cursor=rt.cursor,
            counters=rt.counters,
            rng_manager=rt.rng_manager,
            validation_loss=2.5,
            generated_sample=np.array([1, 2, 3]),
        )
        # A second extraction from the same runtime (probes are deterministic).
        s2 = extract_computational_state(
            model_params=rt.model_params,
            model_buffers={},
            optimizer=rt.optimizer_state,
            scheduler=rt.scheduler,
            cursor=rt.cursor,
            counters=rt.counters,
            rng_manager=rt.rng_manager,
            validation_loss=2.5,
            generated_sample=np.array([1, 2, 3]),
        )
        result = compare(s1, s2)
        assert result["equal"] is True
        assert result["mismatches"] == []

    def test_distinct_states_report_mismatches(self) -> None:
        rt = _runtime_at_k(2)
        s1 = extract_computational_state(
            model_params=rt.model_params,
            model_buffers={},
            optimizer=rt.optimizer_state,
            scheduler=rt.scheduler,
            cursor=rt.cursor,
            counters=rt.counters,
            rng_manager=rt.rng_manager,
            validation_loss=2.5,
            generated_sample=np.array([1, 2, 3]),
        )
        # Perturb one parameter to force a parameters_digest mismatch.
        perturbed = {k: v.copy() for k, v in rt.model_params.items()}
        perturbed["embed"] = perturbed["embed"] + 1.0
        s2 = extract_computational_state(
            model_params=perturbed,
            model_buffers={},
            optimizer=rt.optimizer_state,
            scheduler=rt.scheduler,
            cursor=rt.cursor,
            counters=rt.counters,
            rng_manager=rt.rng_manager,
            validation_loss=2.5,
            generated_sample=np.array([1, 2, 3]),
        )
        result = compare(s1, s2)
        assert result["equal"] is False
        assert "parameters_digest" in result["mismatches"]
        assert "computational_digest" in result["mismatches"]

    def test_comparison_report_uses_closed_diagnostic_domain(self) -> None:
        from expertforge.smoke.report import (
            ComparisonEnvironment,
            ComparisonMismatchField,
        )

        # The Literal domain is a fixed closed set; constructing a report with a
        # mismatch not in the domain must fail with a ValidationError.
        with pytest.raises(ValidationError):
            ComparisonReport(
                u0_run_id="run-x",
                u0_attempt_id="attempt-x",
                r1_run_id="run-y",
                r1_attempt_id="attempt-y",
                specification_fingerprint="spec-v1-sha256-" + "a" * 64,
                n_updates=8,
                k_updates=2,
                tokens_per_update=16,
                u0_computational_digest="x",
                r1_computational_digest="y",
                u0_validation_loss=2.9,
                r1_validation_loss=2.8,
                generated_sample_digest="z",
                decision="fail",
                mismatches=("not_a_real_field",),  # type: ignore[arg-type]
                environment=ComparisonEnvironment(
                    python_version="3.11",
                    numpy_version="2",
                    platform="test",
                    machine="x64",
                ),
            )
        # The domain is non-empty and closed.
        assert ComparisonMismatchField is not None
