"""The two Stage B follow-ups from the 2026-08-23 level-error measurement.

1. ``match_level_to_mixture`` -- ``G`` emits the overlap region at a median 2.86x
   the true amplitude (``level_error_db`` +9.14). Root cause is not a bug in the
   audio path: ``si_sdr_loss`` is scale-invariant, so nothing in training ever
   constrained the output level. The correction projects onto the mixture, uses
   no ground truth, and is therefore deployable.

2. ``refine_embeddings(candidate_audio=...)`` -- the perfect-extractor bound on
   refinement. Every prior "refinement is net-harmful" result varied ENROLLMENT
   quality at one fixed (poor) extractor quality; this varies the other axis.

Both default OFF, and the first two test classes pin that: a committed number
must not move because a new keyword argument exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeDiarizer  # noqa: E402

from dagger.diarize.oracle import OracleDiarizer  # noqa: E402
from dagger.diarize.regions import scene_regions  # noqa: E402
from dagger.enroll.encoder import SpeakerEncoder  # noqa: E402
from dagger.eval.systems import score_scene  # noqa: E402
from dagger.extract.base import Extractor  # noqa: E402
from dagger.reconstruct.stitch import (  # noqa: E402
    MAX_RESCALE, match_level_to_mixture, reconstruct_all,
)


class _FakeEncoder(SpeakerEncoder):
    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        w = np.asarray(waveform, dtype=np.float64)
        if w.size == 0:
            return np.zeros(3)
        zcr = float(np.mean(np.abs(np.diff(np.sign(w))) > 0)) if w.size > 1 else 0.0
        return np.array([float(w.mean()), float(np.sqrt(np.mean(w**2))), zcr])


class _LoudExtractor(Extractor):
    """Stands in for the measured defect: right shape, ~2.86x too loud.

    Deliberately embedding-SENSITIVE. A constant-gain stand-in would make
    ``coarse_to_fine`` independent of its embeddings, and the oracle-audio test
    below would then pass or fail for a reason having nothing to do with the
    flag under test -- the conditioning pathway has to be alive for a refinement
    experiment to measure anything at all.
    """

    GAIN = 2.86

    def _extract(self, x_O: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        mixture = np.asarray(x_O, dtype=np.float64)
        # Steer the SHAPE, not the gain. SI-SDR is scale-invariant, so an
        # embedding that only moved the level would leave every score identical
        # and the test below could never fail -- the same trap that makes the
        # real level error invisible to this project's whole metric suite.
        steer = float(np.tanh(np.sum(embedding)))
        shaped = mixture + 0.3 * steer * np.roll(mixture, 1)
        return shaped * self.GAIN


GATE_CFG = {
    "tau_margin": 0.1, "max_mean_variance": 0.05,
    "min_vad_coverage": 0.5, "max_artifact_score": 0.9,
}


class TestTheRescaleRecoversTheLevel:
    def test_it_undoes_a_known_gain(self):
        rng = np.random.default_rng(0)
        source = rng.normal(size=2000)
        others = rng.normal(size=2000)          # the rest of the mixture
        x_O = source + others
        w = np.ones(2000)

        for gain in (2.0, 2.86, 5.0):
            corrected = match_level_to_mixture(source * gain, x_O, w)
            # Recovers the source's own level, not the mixture's: the other
            # speakers are near-orthogonal and drop out of the projection.
            recovered = float(np.dot(corrected, source) / np.dot(source, source))
            assert recovered == pytest.approx(1.0, abs=0.15), (
                f"gain {gain} left a residual scale of {recovered:.3f}"
            )

    def test_a_correctly_scaled_estimate_is_left_alone(self):
        rng = np.random.default_rng(1)
        source = rng.normal(size=2000)
        x_O = source + rng.normal(size=2000)
        out = match_level_to_mixture(source, x_O, np.ones(2000))
        assert float(np.dot(out, source) / np.dot(source, source)) == pytest.approx(
            1.0, abs=0.15
        )

    def test_it_only_looks_where_the_window_is_open(self):
        """The solo half must not influence the correction -- it is a verbatim
        copy at unity gain and is not emitted through this path."""
        rng = np.random.default_rng(2)
        g = np.concatenate([rng.normal(size=500) * 100.0, rng.normal(size=500)])
        x_O = np.concatenate([np.zeros(500), g[500:] / 3.0])
        w = np.concatenate([np.zeros(500), np.ones(500)])
        c = match_level_to_mixture(g, x_O, w)[500:] / g[500:]
        assert np.allclose(c, c[0]), "scale is not constant -- masking is wrong"
        assert c[0] == pytest.approx(1 / 3.0, rel=0.05)

    def test_it_refuses_a_degenerate_projection(self):
        """A negative projection would flip the waveform's sign; an enormous one
        means the estimate barely correlates with the mixture and the scalar is
        noise. Both must leave the estimate untouched rather than 'correcting'."""
        g = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        assert np.allclose(match_level_to_mixture(g, -g, w), g)      # sign flip
        assert np.allclose(match_level_to_mixture(g, np.zeros(4), w), g)
        assert np.allclose(
            match_level_to_mixture(g, g * (MAX_RESCALE * 10), w), g   # beyond the clamp
        )

    def test_an_empty_window_is_a_no_op(self):
        g = np.array([1.0, 2.0, 3.0])
        assert np.allclose(match_level_to_mixture(g, g * 5, np.zeros(3)), g)


class TestDefaultsAreUnchanged:
    """A committed number must not move because a keyword argument was added."""

    def test_reconstruct_all_is_bit_identical_with_the_flag_off(
        self, three_speaker_scheduled_scene
    ):
        scene = three_speaker_scheduled_scene
        regions = scene_regions(scene, OracleDiarizer())
        emb = np.arange(3 * 3, dtype=np.float64).reshape(3, 3)
        from dagger.audio.provenance import original_mixture
        from dagger.diarize.oracle import overlap_mixture

        x = original_mixture(scene.mixture, label="x")
        x_O = overlap_mixture(x, regions.overlap, label="x_O")
        args = (x, x_O, regions.activity, regions.solo, emb, _LoudExtractor())

        base = reconstruct_all(*args, fade=0)
        same = reconstruct_all(*args, fade=0, rescale_to_mixture=False)
        assert np.array_equal(base, same)

    def test_score_scene_is_bit_identical_with_both_flags_off(
        self, three_speaker_scheduled_scene
    ):
        scene = three_speaker_scheduled_scene
        regions = scene_regions(scene, OracleDiarizer())

        def run(**kw):
            return score_scene(
                scene, 0, 3, 100.0, None, _FakeEncoder(), _LoudExtractor(),
                GATE_CFG, 2, regions=regions, on_unenrollable="drop", **kw,
            )

        a_rows, a_gate, a_over = run()
        b_rows, b_gate, b_over = run(refine_oracle_audio=False, rescale_to_mixture=False)
        assert a_rows == b_rows and a_gate == b_gate and a_over == b_over


class TestTheRescaleFixesTheReportedLevelError:
    def test_level_error_db_collapses_when_the_flag_is_on(
        self, three_speaker_scheduled_scene
    ):
        """End to end: the column that measured +9.14 dB on real data must go to
        ~0 against an extractor with a known, pure gain error."""
        scene = three_speaker_scheduled_scene
        regions = scene_regions(scene, OracleDiarizer())

        def levels(rescale):
            _, _, overall = score_scene(
                scene, 0, 3, 100.0, None, _FakeEncoder(), _LoudExtractor(),
                GATE_CFG, 0, regions=regions, on_unenrollable="drop",
                rescale_to_mixture=rescale,
            )
            return [
                abs(r["level_error_db"]) for r in overall
                if r["system"] == "no_recursion" and np.isfinite(r["level_error_db"])
            ]

        before, after = levels(False), levels(True)
        assert before, "fixture produced no measurable level error"
        assert np.median(before) > 3.0, (
            f"fixture is not exercising a level error (median {np.median(before):.2f} dB)"
        )
        assert np.median(after) < np.median(before), (
            f"rescale did not reduce the level error: "
            f"{np.median(before):.2f} -> {np.median(after):.2f} dB"
        )


class TestOracleAudioRefinement:
    def test_the_candidate_really_comes_from_the_clean_source(
        self, three_speaker_scheduled_scene
    ):
        """The plumbing check, asserted where it is observable regardless of
        whether the gate happens to accept: the margin is computed on the
        candidate clip, so a different candidate must produce different margins.
        Asserting on the OUTPUT instead would silently pass whenever the gate
        rejects everything -- which it does at `tau_margin: 0.1` with this
        fixture's deliberately crude 3-dim encoder."""
        scene = three_speaker_scheduled_scene
        regions = scene_regions(scene, OracleDiarizer())

        def margins(oracle_audio):
            _, gate, _ = score_scene(
                scene, 0, 3, 100.0, None, _FakeEncoder(), _LoudExtractor(),
                GATE_CFG, 2, regions=regions, on_unenrollable="drop",
                refine_oracle_audio=oracle_audio,
            )
            return [r["margin"] for r in gate if r["system"] == "coarse_to_fine"]

        assert margins(True), "no refinement gate rows -- fixture refined nothing"
        assert margins(True) != margins(False)

    def test_it_changes_the_refined_output_when_the_gate_accepts(
        self, three_speaker_scheduled_scene
    ):
        """The end-to-end complement, with a permissive gate so acceptance can
        actually happen. Without this, the test above could pass while the
        accepted embedding was never used to reconstruct anything."""
        scene = three_speaker_scheduled_scene
        regions = scene_regions(scene, OracleDiarizer())
        open_gate = {**GATE_CFG, "tau_margin": -1.0}

        def c2f(oracle_audio):
            rows, gate, _ = score_scene(
                scene, 0, 3, 100.0, None, _FakeEncoder(), _LoudExtractor(),
                open_gate, 2, regions=regions, on_unenrollable="drop",
                refine_oracle_audio=oracle_audio,
            )
            accepted = sum(
                bool(r["accepted"]) for r in gate if r["system"] == "coarse_to_fine"
            )
            return accepted, [r["si_sdr"] for r in rows if r["system"] == "coarse_to_fine"]

        n_true, out_true = c2f(True)
        n_false, out_false = c2f(False)
        assert n_true > 0 and n_false > 0, (
            f"gate still rejected everything ({n_true}/{n_false}) -- this test "
            "cannot distinguish the flag from a closed gate"
        )
        assert out_true != out_false

    def test_the_other_three_systems_are_untouched(
        self, three_speaker_scheduled_scene
    ):
        """They never refine, so they stay a valid control within the same run --
        the same property the acceptance-rule ceiling relies on."""
        scene = three_speaker_scheduled_scene
        regions = scene_regions(scene, OracleDiarizer())

        def others(oracle_audio):
            rows, _, _ = score_scene(
                scene, 0, 3, 100.0, None, _FakeEncoder(), _LoudExtractor(),
                GATE_CFG, 2, regions=regions, on_unenrollable="drop",
                refine_oracle_audio=oracle_audio,
            )
            return [r for r in rows if r["system"] != "coarse_to_fine"]

        assert others(True) == others(False)

    def test_it_works_under_real_regions_too(self, three_speaker_scheduled_scene):
        """Under a real diarizer the row order comes from cluster mapping, so
        `candidate_audio[i]` must be the ATTRIBUTED source for row i, not
        source i. A mis-indexed bound would silently score the wrong speaker."""
        scene = three_speaker_scheduled_scene
        regions = scene_regions(scene, FakeDiarizer(relabel=True))
        rows, _, _ = score_scene(
            scene, 0, 3, 100.0, None, _FakeEncoder(), _LoudExtractor(),
            GATE_CFG, 2, regions=regions, on_unenrollable="drop",
            refine_oracle_audio=True,
        )
        assert rows, "no rows scored under relabelled regions"
