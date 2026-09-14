# Experiment Tracker — M-10 Arrival GPPO Candidate Fusion

| Block | Status | Evidence |
|---|---|---|
| Actual HEAD/worktree check | DONE | HEAD `0396c7b3976b4ac71f7d4953e377ecf5a2f71b31`; unrelated untracked release/pilot files preserved |
| Arrival protocol boundary | DONE | `docs/contracts/m10-arrival-gppo-fusion-interface-v1-20260914.md` |
| Candidate-wise scorer and frozen model loader | DONE | `gppo_world/arrival_gppo_fusion.py` |
| Candidate prior and legal action selection | DONE | `CandidateAwarePolicy`, `choose_candidate_action`, fixed mask/tie behavior |
| Public-observation to environment-step fixture | DONE | `tests/test_arrival_gppo_fusion.py` |
| Interface/arrival targeted tests | DONE | 12 passed |
| New policy PPO fusion runner | NOT STARTED | Deliberately not fabricated in this design/interface revision |
| New data or model training | NOT RUN | Explicitly outside this round |
| Single-seed fusion pilot | NOT RUN | Requires compatible arrival policy/model artifacts and a new runner |
| Multi-seed fair comparison | NOT RUN | Conditional on pilot gate |

No old budget is reset. Historical releases, negative results, and untracked local artifacts remain unchanged.
