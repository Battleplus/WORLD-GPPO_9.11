# Experiment Plan — M-10 Joint Task-Set Consequences

**Status:** design and interface only; no data generation or training run in this revision.

## Claim map

| Claim | Evidence required | Stop if |
|---|---|---|
| Joint labels expose action opportunity cost | Same-prefix candidate differences persist across paired repeats, with fixed task denominator and valid masks | all labels censored or repeat variance dominates |
| A learned predictor adds ranking value | It beats the frozen public joint score on held-out parent groups | public score explains ranking or no ranking gain |
| A prediction prior improves scheduling | Independent parent tapes show better selected joint outcomes with no physical-arrival/safety/cost regression | no selected-branch gain, safety issue, or budget limit |

## Frozen design

- Primary outcome: on-time physical arrivals in the public unfinished task set.
- Secondary outcome: observed deadline failures and candidate-minus-reference paired differences.
- Continuation: candidate once, then `traditional_public_controller_v1`.
- Reference: predeclared legal current rule; never a post-hoc best branch.
- Splits: parent episode is the unit; no parent/prefix/repeat crosses split.
- Development pilot: 8 train parents, 4 validation parents, four prefixes, up to 25 candidates, three exogenous repeats.
- Conditional formal data: 64/32/64/32 parents for train/validation/test/OOD.
- Conditional model pilot: one seed 1101, 512 updates, 4 epochs, patience 2, 20 minutes.

## Explicit exclusions

No notification change, event trigger, reward shaping, total-system-energy residual route, old arrival-time checkpoint reuse, or large-scale generation is authorized by this design-only revision.
