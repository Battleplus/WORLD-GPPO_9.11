# Experiment Tracker — M-10 Joint Task-Set Consequences

| Block | Status | Evidence |
|---|---|---|
| Protocol/schema frozen | DONE | `docs/world-model/m10-joint-consequence-protocol-20260914.md`; schema file |
| Existing generator adapter | DONE | `tools/generate_m10_joint_consequence_dataset.py`; not run in this revision |
| Public joint baseline | DONE | `gppo_world/joint_consequence_baseline.py`; fixture test |
| Fixed fixture contract tests | DONE | `tests/test_joint_consequence.py` |
| Development data generation | DONE | v4: 600 train / 243 validation records; 843 branches; manifest audit passed |
| Model training | DONE | CUDA seed 1101; 152 updates; completed 4 epochs; development evidence only |
| Formal test/OOD | NOT RUN | No formal data generated |
| Online GPPO fusion | NOT RUN | Conditional on independent parent-tape ranking evidence |

Historical arrival-time/system-energy results and notification candidate results remain in their original versions and are not relabeled as evidence for this hypothesis.
