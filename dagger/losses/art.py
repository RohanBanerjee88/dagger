"""L_art -- the artifact term. NOT IMPLEMENTED: no formula exists anywhere.

A bare symbol in both documents. "Artifact" appears exactly twice in
docs/diarization_full_mathematical_theory.pdf -- S9 Eq. (16)'s gate score and
S10's objective -- and is defined in neither. The errata repeats both (S5
Eq. 11, S6), spells it L_artifact, and defines it in neither. BOTH documents
have been checked; there is nothing to look up. (Speech(s_hat_i), in the same
gate score, is in the same position.)

TWO TESTS ANY CANDIDATE MUST PASS. The second was learned the hard way.

1. VANISH AT THE TRUTH (Proposition 7 / Corollary 1). This kills the naive
   form: L_art = F(s_hat_i) for F = spectral flatness drives F -> 0, but clean
   speech measures F ~ 0.37, so minimising it moves the estimate AWAY from the
   target.

2. ORDER DAMAGED ESTIMATES MONOTONICALLY against SI-SDR.

THE CANDIDATE THAT PASSED (1) AND FAILED (2), measured 2026-08-28:

    L_art = ( F(s_hat_i * w_Oi) - F(s_i * w_Oi) )^2

Consistency is sound -- exactly 0 at the truth, Corollary 1's repair reapplied.
Scored against one speech-like target, F(s_i) = 0.3268:

    estimate              SI-SDR       F        candidate
    good extraction       +17.0 dB   0.5836     0.06592
    broadband hiss         +4.9 dB   0.7495     0.17861
    80% spectral holes    -37.8 dB   0.3078     0.00036

A destroyed signal is charged 180x LESS than a good extraction: the term has a
pathological minimum and would prefer that G wreck the signal via spectral
holes over extracting it well. F is hypersensitive upward (noise fills spectral
valleys; the geometric mean is dominated by its smallest bins) and nearly blind
downward (overlap-add refills punched holes), so it tracks the broadband noise
floor -- which L_sep already penalises -- and misses the structural corruption
L_art was meant to add value on. Same blindness Stage B Session 4 found in the
same diagnostic as a gate check (0.7376 on clean audio vs 0.7422 on G's
output), one layer up and now with a sign that rewards damage.

If a flatness-based candidate is chosen anyway, make it TWO-SIDED, not a
one-sided hinge: relu(F(s_hat) - F(s))^2 assumes every artifact RAISES
flatness, and over-suppression lowers it (S5: a network on clean audio "can
only add artifacts (over-suppression, phase distortion)"). Two-sided also
decouples this term from phase3_gate_faults.yaml, which is the run that would
measure the direction.
"""

from __future__ import annotations


def art_loss(*args, **kwargs):
    raise NotImplementedError(
        "L_art has no formula in either theory document, and the one candidate "
        "derived for it was measured and rejected -- it charges a destroyed "
        "signal 180x less than a good extraction. See this module's docstring "
        "for the two tests any replacement must pass."
    )
