# Experiment Tracker — M-10 Arrival GPPO Candidate Fusion

| Block | Status | Evidence |
|---|---|---|
| Actual HEAD/worktree check | DONE | HEAD `0396c7b3976b4ac71f7d4953e377ecf5a2f71b31`; unrelated untracked release/pilot files preserved |
| Arrival protocol boundary | DONE | `docs/contracts/m10-arrival-gppo-fusion-interface-v1-20260914.md` |
| Candidate-wise scorer and frozen model loader | DONE | `gppo_world/arrival_gppo_fusion.py` |
| Candidate prior and legal action selection | DONE | `CandidateAwarePolicy`, `choose_candidate_action`, fixed mask/tie behavior |
| Public-observation to environment-step fixture | DONE | `tests/test_arrival_gppo_fusion.py` |
| Interface/arrival targeted tests | DONE | 12 passed |
| New policy PPO fusion runner | DONE | `tools/run_m10_arrival_gppo_fusion_pilot.py`; sampling and update log-probabilities use the same candidate-aware distribution |
| New data or model training | NOT RUN | No new consequence data/model training; frozen arrival model only |
| Single-seed fusion pilot | DONE | Local run `m10-arrival-gppo-fusion-pilot-20260914-seed1101`; 512 steps and 8 updates per learning group |
| Checkpoint reload and first-decision verification | DONE | `tools/verify_m10_arrival_gppo_fusion_pilot.py`; all three learning checkpoints passed |
| Multi-seed fair comparison | NOT RUN | Conditional on pilot gate |

No old budget is reset. Historical releases, negative results, and untracked local artifacts remain unchanged.
