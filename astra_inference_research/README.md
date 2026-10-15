# Astra inference research — dated snapshots

This repository separates the original 2026-10-14 research scaffold from the
2026-10-15 audited continuation. The old GitHub scaffold is archived unchanged.

| Folder | Meaning |
| --- | --- |
| [101401/](101401/) | **Exact original GitHub version** moved from this directory; original files preserved |
| [101501/](101501/) | **Audited 2026-10-15 continuation**: new layer-specific residual schedule, safe per-chunk refresh, strict diagnostics/tests |

The uploaded research archive `26101501.zip` (2026-10-15) was reviewed to
design the continuation. It contains intermediate cache experiments and logs; the uploaded ZIP contains **86 completed metadata files and no MP4 videos because the user deliberately omitted the already-generated videos to reduce upload size**. The videos exist outside the ZIP and are **not automatically copied into GitHub**. See
[101501/docs/AUDIT.md](101501/docs/AUDIT.md) for provenance, evidence,
limitations, and a detailed list of what was used or intentionally omitted.

For new work begin with [101501/README.md](101501/README.md).

**No world-model re-training; no claims of GPU validation of the new 101501 code.**


## 2026-10-16: 101601 restores token-level scope

- [101601/](101601/) is a NEW research branch built from the actual seven-page
  Dual-Horizon AdaptiveCache proposal and the 26101502 feedback.
- It is not another whole-block K/interval schedule. It implements per-token
  selected-query compute, FramePack token grouping, token gate studies, and
  one-token future rollout counterfactual training-data interfaces.
- [101601 docs/RESEARCH_PLAN.md](101601/docs/RESEARCH_PLAN.md) records the
  exact scientific scope and honest limitations.
- 101501 remains the block-level baseline, not a prerequisite to 101601.
- GPU inference, full Astra numerical equality, teacher-trained gate
  generalization and end-to-end speedup are **NOT YET VALIDATED**.
