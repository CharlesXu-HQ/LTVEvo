# ModelEvoHarness alignment

## Decision

LTVEvo will use the same boundary as CouponEvo: ModelEvoHarness supplies a versioned research catalog, task applicability, bounded reference reading, and proposal/reflection checks. LTVEvo remains the host for data preparation, candidate execution, validation metrics, exact-dataset experience, and one-time final test evaluation. This preserves the fixed-horizon prediction task and its GPU sandbox.

ModelEvoHarness 0.2.0 has no LTV prediction stage or bundled LTV regressor. The LTV adapter declares `stage=ltv_prediction` and contributes one local research family through the documented `extra_families` API. It does not claim that a ranking, policy, or uplift method is suitable for the regression target. LTV reference candidate source is host-owned material with a host-computed SHA-256; it is not described as a bundled Harness reference.

## Contract

- `--harness model-evo` starts a new search with a pinned ModelEvoHarness submodule. Existing searches retain their current mode and remain readable. A changed Harness commit, catalog, implementation, LTV adapter, or frozen dataset/task identity prevents resume.
- The adapter builds a task snapshot from the prepared metadata and training partition only: exact raw and split hashes, positive-purchase target, horizon, observation dates, actual feature names, `MAE/min`, and `pytorch`. Test rows and final metrics never enter proposal context.
- Agent proposals state a falsifiable `research` hypothesis with real input fields, one mechanism, a same-validation comparison, expected result, falsification condition, and an alternative. Harness validates the research contract. The Agent may read bounded bundled reference modules through `propose_with_references` before writing self-contained PyTorch candidate code.
- The host evaluates only the frozen validation split in its existing no-network Docker GPU sandbox. Reflection records a technical lesson and an explicit `business_experience: not_observable`; prediction errors cannot establish coupon uplift or business impact. Existing `high` and anomaly-triggered `max` behavior remains.
- The search journal records Harness identity, task snapshot, research, reference hashes, candidate hash, measured metrics, reflection, and dataset-bound experience. The existing champion rule and one-time final test/paired interval are retained.

## Acceptance

1. An integration test proves the LTV snapshot marks the LTV family ready and policy/ranking families inapplicable, and the identity changes with the pinned Harness implementation or dataset fingerprint.
2. An Agent search with a fake provider uses Harness research validation and bounded source reading, preserves the sealed test and resume behavior, and cannot promote an invalid proposal.
3. The English and Chinese READMEs show the same optional install and CLI mode. Full-data validation on the designated GPU machine runs without sampling, and the report separates workflow success from model-quality improvement.

## Out of scope

Changing ModelEvoHarness upstream, migrating the whole search journal to its `run_search` engine, causal/uplift tasks, or treating an LTV prediction metric as an intervention effect.
