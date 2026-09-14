# Experiment Tracker — M-10 Joint Task-Set Consequences

| Block | Status | Evidence |
|---|---|---|
| Protocol/schema frozen | DONE | `docs/world-model/m10-joint-consequence-protocol-20260914.md`; schema file |
| Existing generator adapter | DONE | `tools/generate_m10_joint_consequence_dataset.py`; not run in this revision |
| Public joint baseline | DONE | `gppo_world/joint_consequence_baseline.py`; fixture test |
| Fixed fixture contract tests | DONE | `tests/test_joint_consequence.py` |
| Development data generation | NOT RUN | Explicitly deferred by scope |
| Model training | NOT RUN | Requires data audit and a new schema-specific trainer |
| Formal test/OOD | NOT RUN | No formal data generated |
| Online GPPO fusion | NOT RUN | Conditional on independent ranking evidence |

Historical arrival-time/system-energy results and notification candidate results remain in their original versions and are not relabeled as evidence for this hypothesis.
