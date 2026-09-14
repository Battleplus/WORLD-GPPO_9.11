# Experiment Plan — M-10 Arrival GPPO Candidate Fusion

This version records the next-stage executable design and interface boundary.

- Starting HEAD verified: `0396c7b3976b4ac71f7d4953e377ecf5a2f71b31`.
- Protocol: `world-gppo-9.11-arrival/0.1.0`; primary metric `physical_arrival`.
- Comparators: legal public rule, GPPO, GPPO-History, and GPPO-History plus candidate-wise arrival consequence prior.
- No notification change, event trigger, reward shaping, old continuous-service checkpoint reuse, new data generation, or training was run in this revision.
- Interface: `gppo_world/arrival_gppo_fusion.py`; targeted contract/chain tests passed.
- Candidate feature order is frozen and never averaged across action slots.
- Proposed pilot: seed 1101; 512 environment steps per learning group; 128 optimizer updates; 16 fixed validation tapes; 30 minutes per group; frozen model and zero extra model data.
- Stop on leakage, safety violation, non-finite state, incomplete terminal accounting, resource shortage, or any bound.

The existing joint-consequence checkpoint is not protocol-compatible with the arrival model interface. A compatible arrival policy/model artifact with recorded hashes is a prerequisite for a future pilot; this document does not claim that artifact exists.
