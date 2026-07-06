"""Enrollment: top-K solo clips -> phi -> mean embedding ``e_bar_i`` + variance ``V_i``.

``dagger.enroll.encoder`` holds phi (:class:`~dagger.enroll.encoder.TitaNetEncoder`,
NVIDIA NeMo's TitaNet-Large, frozen); ``dagger.enroll.topk`` selects the top-K
solo clips per speaker and builds the mean embedding + variance. ``V_i`` is
computed now but unused until Phase 3's enrollment-rejection gate. See CLAUDE.md §5.
"""
