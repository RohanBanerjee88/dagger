"""L_sep -- the separation term. NOT IMPLEMENTED: the theory gives no formula.

S10 says only "with L_sep driving s_hat_i -> s_i"; errata S6 says "a separation
term pushing s_hat_i -> s_i". Neither document writes the term out or names a
metric. There is nothing here to transcribe.

WHAT THE CODE ACTUALLY DOES, and why it is not this module: training calls
``dagger.losses.sisdr.si_sdr_loss`` directly (scripts/train_phase1.py:223).
SI-SDR was chosen in Phase 1 to mirror ``dagger.metrics.sisdr``, which landed
two days earlier in Phase 0. That is a pragmatic choice of OUR own, and it
stays at the call site so that no module here can be mistaken for the theory's
L_sep.

THE CHOICE IS LOAD-BEARING. SI-SDR is scale-invariant, so it cannot constrain
output level. It has been the only term the code ever had, and with L_recon
dropped, nothing constrained level at all: G emitted the overlap region at
2.86x the true amplitude for three phases (CLAUDE.md S9).

If a formula is ever settled, it lands here and the call site moves to it.
Scale-dependent SNR -- the same ratio without the target rescaling -- is the
obvious candidate. DO NOT adopt it in the same run that lands new terms:
Stage C predicts level_error_db collapses when L_recon arrives, and changing
L_sep simultaneously makes that result unattributable.
"""

from __future__ import annotations


def sep_loss(*args, **kwargs):
    raise NotImplementedError(
        "L_sep has no formula in either theory document -- S10 and errata S6 "
        "describe it only as 'a separation term pushing s_hat_i -> s_i'. "
        "Training deliberately calls dagger.losses.sisdr.si_sdr_loss directly; "
        "see this module's docstring for why that is not the same claim."
    )
