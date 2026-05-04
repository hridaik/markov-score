# PROGRESS.md

## Current phase: Phase 0 (Ground Truth) — NOT STARTED

## Phase status

| Phase | Status | Key numbers | Blocker |
|-------|--------|-------------|---------|
| 0A — Lyapunov solution | Not started | — | — |
| 0B — Fokker-Planck numerics | Not started | — | Depends on 0A |
| 1A — Simulation infrastructure | Not started | — | Depends on 0A |
| 1B — Score matching | Not started | — | Depends on 1A |
| 1C — Graphical lasso | Not started | — | Depends on 1A |
| 1D — Method comparison | Not started | — | Depends on 1B, 1C |
| 2A — Bifurcation sweep | Not started | — | Depends on Phase 1 |
| 2B — Mixture graphical model | Conditional | — | Depends on 2A results |
| 3A — Jacobian estimation | Not started | — | Depends on 1A |
| 3B — J-vs-H comparison | Not started | — | Depends on 3A, Phase 1 |
| 3C — Blanket phase transition | Conditional | — | Depends on 3B, 2A |
| 4A — Subsampling sweep | Not started | — | Depends on Phase 1 |
| 5A — Summary figures | Not started | — | Depends on all above |
| 5B — Conclusions | Not started | — | Depends on 5A |

## Next action on resume
Begin Phase 0A: implement Lyapunov solver and compute ground truth H(κ).

## Notes for supervisor
Project initialised. No code written yet.