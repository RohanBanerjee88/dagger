# Phase 1 — Identity conditioning: the working record

Moved verbatim out of `CLAUDE.md` on 2026-08-25 (relocation only, no edits). The
**plan** and the **DoD verdict** stay in CLAUDE.md §5; this is how that verdict was
reached — the failed runs, the diagnoses, and the fixes.

Read before: retraining `G`, or changing the loss / crop sampling.

---

**⚠ KNOWN ISSUE (found 2026-07-11, first Phase 1 runs): `stagger_offsets` starves speakers of
solo time on 3+ speakers.** Chain placement starts utterance *i+1* at `(1 − overlap)` into
utterance *i* (`dagger/data/mixing.py`), so solo time depends on *random length ratios*: with
`overlap: 0.5`, the middle speaker of a 3-mix gets a solo window only if `L2 > L1`, and the last
only if `L3 > ~0.5·L2`. Result on Libri3Mix: **~70–80% of scenes are skipped at enrollment**
(`NoSoloRegionError`, caught and logged — not a crash), silently shrinking the effective
training/eval sets and biasing survivors toward `L2 > L1` orderings. The mixing docstring's
promise ("every mixture has a solo lead-in, an overlap middle, and a solo tail") only holds for
2 speakers. The skip-on-no-solo behavior itself is correct and stays (unenrollable speakers are
real; Phase 3 must handle them) — the bug is that our own generator manufactures them.
*Fixed (2026-07-11):* `stagger_offsets` now takes `min_solo` (samples) and pushes each start
just late enough that every speaker keeps a contiguous solo window of
`min(min_solo, own length)`; both loaders pass it via the `min_solo_ms` config key
(default 1000 ms — above enrollment's 500 ms `min_clip_ms`). `min_solo_ms: 0` restores the
legacy length-ratio-dependent behavior. Trade-off: the guarantee takes precedence over
`overlap`, so adjacent short utterances may overlap less than requested.
*Phase 2 heads-up:* a chain-staggered scene where the middle speaker has solo time **cannot**
contain a depth-3 overlap (s2's solo requires s1 to end before s3 starts). The depth-stratified
experiment needs both per-speaker solos *and* deep overlaps, so Phase 2 placement must become a
small scheduler (e.g., per speaker: one guaranteed solo segment + one deliberately deep
overlapped segment) — the solo-aware offset fix above is not sufficient for Phase 2.

**⚠ KNOWN ISSUE (found 2026-07-11, first full training run): degenerate loss terms drowned the
training signal for `G`.** Symptoms: training loss flat at ~40 (i.e. −40 dB SI-SDR — worse than
outputting the mixture, impossible for a real comparison) for 25 epochs, and proposed vs blind
eval rows agreeing within ~0.3 dB — two different architectures both stuck near passthrough.
Cause: uniform-random 4 s training crops usually miss a given speaker's overlap window, and
SI-SDR is *scale-invariant*, so a ~zero windowed target can't express "output silence" — the
term degenerates to `−10·log10(eps)` (~+80) with a garbage gradient that swamps the scoreable
terms. The `min_solo` fix made this *more* common (it reduces overlap by design). Same disease,
milder, in the blind system's PIT loss (silent speaker in crop).
*Fixed (2026-07-11):* (a) `scripts/train_phase1.py` masks out (crop, speaker) terms whose
windowed target has ~zero energy and averages over scoreable terms only (skipped terms also
skip their forward pass); (b) `dagger/losses/pit.py` masks silent target speakers out of the
per-item mean and drops all-silent items; (c) `dagger/data/torch_adapter.py` centers each crop
on a random overlap sample (uniform over overlap samples via precomputed run boundaries;
uniform-start fallback when a scene has no overlap), and `require_overlap=True` (used by
proposed training) drops zero-overlap scenes with a logged count — such scenes still exist
because the `min_solo` guarantee can push short utterances fully clear of the chain; at eval
they appear as `overlap: n/a` rows (speaker is 100% copy-path), which is correct behavior.
Healthy-training signature going forward: loss starts ~5–15 and *trends down*; a flat loss
near +40 means degenerate terms are back.

**⚠ OPEN ISSUE (found 2026-07-11, second full training run, after the loss fixes): the proposed
extractor collapses to passthrough — its embedding conditioning is not being used.** Evidence:
with the fixed losses, the **blind** baseline now trains cleanly (loss 0.36 → −2.26 over 25
epochs; eval overlap SI-SDR 0.01 → 2.26 dB), but the **proposed** system's loss oscillates
around 0.1–0.7 with no trend, and its eval `overlap(prop)` column is identical (±0.01 dB)
across two independently trained runs — only possible if both runs output ≈ a scaled copy of
`x_O` (SI-SDR is scale-invariant, so every `c·x_O` scores exactly what the mixture scores).
Mechanism: one output head is asked for three different answers from the *same* input,
disambiguated only by `ē_i`; if the conditioning pathway is too weak to matter early, the three
per-speaker gradients on identical input cancel and "output the mixture" is a stable resting
point (the blind system escapes because its 3 PIT-matched heads can specialize without
conflicting gradients). No wiring bug found in `extract/tfgridnet_crossattn.py` /
`extract/crossattn.py` / `extract/tfgridnet.py` — the suspicion is *dosage*, not design:
config injects the fusion before only **1 of 6** blocks (`cross_attn_blocks: 1`; the backbone
already supports fusing before every block), the 192-d embedding is compressed to only
`n_tokens: 4` key/value tokens, and raw (unnormalized) TitaNet embeddings feed `token_proj`
(early noisy FiLM scales incentivize the optimizer to mute the pathway).
*Next actions (diagnose before spending GPU):* (1) embedding-sensitivity probe on the saved
checkpoint — `G(x, e_A)` vs `G(x, e_B)` vs `G(x, random)`; near-identical outputs confirm the
collapse; (2) overfit-4-scenes test — if proposed cannot drive its loss strongly negative even
when memorizing, the conditioning pathway is underpowered; (3) if confirmed, remedies in order:
`cross_attn_blocks: 6` (one YAML line), L2-normalize the embedding before `token_proj`,
`n_tokens: 8`, then retrain. Note the current blind-beats-proposed table is **not** a DoD
verdict: blind gets oracle best-permutation matching, and proposed's 0.13 dB is a passthrough
artifact, not a measurement of a working extractor — the Phase 1 comparison hasn't actually
been run yet.
*Diagnosed (2026-07-12): the architecture is fine — the failure is optimization (passthrough
is a plateau the optimizer must escape, and at lr 1e-3 it never does).* Evidence chain:
(a) the dosage remedies (`cross_attn_blocks: 6`, L2-norm in the fusion module, `n_tokens: 8`)
were applied and a fresh capped run (400 scenes / 25 epochs) *still* landed at 0.14 dB overlap —
third run within 0.01 dB of passthrough; (b) the embedding-sensitivity probe
(`scripts/probe_phase1_conditioning.py`) on that checkpoint showed the pathway ALIVE but not
steering (outputs change ~5% when swapping embeddings; SI-SDR vs `x_O` = 35.8 dB ≈ scaled
mixture copy; diag−offdiag margin 0.16 dB ≈ 0); (c) the overfit-4-scenes run
(`configs/phase1/experiments/phase1_overfit4_diag.yaml`, lr 3e-4, 600 single-batch epochs) sat at the passthrough
plateau for ~170 steps, then **escaped**: final loss ~−2 to −3, and the probe on its checkpoint
returned STEERS (outputs change 86% across embeddings, passthrough down to 8.1 dB, diag
+2.38 dB vs offdiag −3.04 dB — pointing G at speaker j actively suppresses speaker i).
Structural remedies (aux speaker-consistency loss, silent-target energy terms, mixture
dropout) are NOT needed on current evidence. *Remedy applied (2026-07-12):* `lr: 3e-4` in the
train config and gradient clipping (`train.grad_clip`, default max-norm 5.0, both systems) in
`scripts/train_phase1.py` — the overfit log showed single unclipped steps (+1-to-+2 loss
spikes) repeatedly erasing hundreds of steps of descent. Retrain pending. Expect a plateau
phase near ~0.3–0.5 loss before escape; a run has failed only if it is still flat at the END,
not because it starts flat.
*RESOLVED (2026-07-13): Phase 1 DoD met.* The lr 3e-4 + grad-clip retrain escaped the plateau
(400 scenes/25 epochs: 1.75 dB vs blind's 1.03; probe on *test* scenes: STEERS). Scaled runs
(Kaggle batch, one T4 each: `limit: 2000, epochs: 30, batch 4, lr 3e-4, grad_clip 10`;
`torch_adapter` now stores crops compactly — float32 audio + uint8 masks, ~3× less host RAM —
after the prepared-scenes list OOMed ~30 GB at 2000 scenes) produced the DoD table
(150 test scenes, 450 rows):
**proposed 4.40 dB vs blind 2.05 dB mean overlap SI-SDR (+2.35 dB)**; probe: passthrough
2.89 dB, diag +5.80 vs off-diag −6.95 (12.8 dB steering margin — grew with data: 5.6 dB at
400 scenes).
*What that means on one recording:* take a 3-speaker mixture and ask for speaker A. The blind
baseline outputs three tracks and you must guess which is A; the proposed system is told "A" via
`ē_A` and returns A's track directly, 2.35 dB cleaner in the overlapped part. The steering probe is
the check that the conditioning is real rather than decorative: pointing `G` at speaker B's
embedding while asking for A's audio makes the output **worse by 6.95 dB**, i.e. it actively
suppresses A — a system ignoring its embedding could not do that.
*Verified against the committed CSV 2026-08-25:* `results/phase1/phase1_librimix_3spk.csv`,
396 scoreable rows of 450, `overlap_proposed` 4.40, `overlap_blind` 2.05, paired +2.35, win 50%,
paired sd 7.52. Caveats recorded honestly: (a) per-row win rate is only 50% (paired std 7.51 dB) —
the mean margin comes from magnitude asymmetry (proposed's wins are much larger than its
losses); Phase 2's depth stratification should locate where the big wins live; (b) both
systems are undertrained (2000 of ~34k Libri3Mix train-360 recipes; loss still descending at
cutoff) so all numbers are lower bounds; (c) the 2-speaker WSJ0-2mix literature-bar check is
deferred — no LDC license — substitute Libri2Mix if ever needed. Reproduce: train both systems
with `configs/phase1/dod/phase1_librimix_3spk_train.yaml` (`--system proposed|blind`), eval with
`scripts/run_phase1.py --config configs/phase1/dod/phase1_librimix_3spk_eval.yaml` (limit 150).
