"""Training losses: SI-SDR, PIT (blind baseline), and the theory's four-term objective.

The theory's objective (docs/diarization_full_mathematical_theory.pdf S10;
errata S6) is::

    L = l1*L_sep + l2*L_spk + l3*L_recon + l4*L_art

Of those four, the documents write out exactly ONE: ``L_recon``. ``L_sep`` is
described only in prose ("a separation term pushing s_hat_i -> s_i"), and
``L_spk`` and ``L_art`` are bare symbols. The modules in this package mirror
that state, not the code's behaviour: ``sep``, ``spk`` and ``art`` raise
NotImplementedError and carry the reasons, ``recon`` is implemented.

What training actually runs is ``dagger.losses.sisdr.si_sdr_loss``, called
directly from scripts/train_phase1.py. SI-SDR was chosen in Phase 1 to mirror
the evaluation metric (``dagger.metrics.sisdr``, Phase 0, two days earlier).
It is deliberately NOT wrapped as ``L_sep``: that would assert a correspondence
the theory never states.

``dagger.losses.pit`` implements Eq. (7), the blind-separation baseline. It is
not a term in the objective -- and it is the only loss here transcribed from a
formula the documents actually write out.
"""
