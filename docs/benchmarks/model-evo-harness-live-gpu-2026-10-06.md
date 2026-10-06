# Live Agent + ModelEvoHarness: full-data GPU validation

Date: 2026-10-06. This run used the real DeepSeek API and the complete frozen Online Retail II dataset. It verifies two Agent-led model iterations under the pinned Harness integration. **Neither candidate improved the primary validation metric.** No final test was run for this journal: these public test rows were inspected in earlier experiments and cannot provide a new independent confirmation.

## Frozen inputs and execution

- Dataset: the complete 1,067,371-row Online Retail II mirror ZIP, SHA-256 `e2a0ebc53c3b0b577ff2f3dc4eb64032a38e38f6d7a47f73e511cdcd6d5b7dbd`. The unchanged prepared snapshots contain 18,472 train, 13,292 validation, and 15,433 sealed test customer-observation rows; see the [full-data preparation record](online-retail-ii-gpu-2026-10-04.md).
- Frozen task fingerprint: `44c020de3029426b3919ba869633c2c092d8a2de1cbbd74fde1cc62bcef93e61`; objective: 90-day positive purchase amount, validation MAE minimized. The journal retains exact raw and split hashes and evaluator version `1`.
- Code: LTVEvo `fe0d5c416d8f771361f87f2596872b35f780e598` (the functional commit later published in `main`), ModelEvoHarness pinned at `23c947ddc90b0a6ebafddfebad228021f1a54d5e`. Harness was not modified.
- Machine: NVIDIA GeForce RTX 5090; host Python 3.12.13 and PyTorch 2.11.0+cu128; candidates executed in the existing `ltvevo-sandbox` CUDA image.
- Provider: [DeepSeek V4.1 Flash](https://api-docs.deepseek.com/news/news260910/) via `deepseek-flash`, with `thinking=enabled` and `reasoning_effort=high` for proposals and reflections, consistent with the [official thinking-mode contract](https://api-docs.deepseek.com/guides/thinking_mode/). A small authenticated JSON request succeeded before the run. The API key existed only in the interactive process environment, was cleared afterwards, and is absent from the journal and repository. No `max` anomaly review was triggered in this run.
- Journal: `/home/charles/LTVEvo-data/runs/retail-90d-harness-live/journal.json` on the GPU host. Search used a fresh journal. `--steps 1` ran the first experiment; resuming with `--steps 2` ran exactly one more. The second proposal saw the first candidate's hypothesis, measured result, and reflection. The Agent also read seven same-task/dataset validation lessons from earlier finalized journals, including the deterministic Harness smoke run; no historical test metrics entered its context.

The run used `ltvevo search --harness model-evo --device cuda --docker-image ltvevo-sandbox --provider-url https://api.deepseek.com --model deepseek-flash --thinking enabled` with the existing frozen task and snapshots. Candidate code, hashes, research records, paired intervals, and reflections remain in the journal. Search did not read test rows or call `finalize`.

From the pinned checkout on the GPU host, with `LTVEVO_API_KEY` set in the process environment, the two calls were:

```bash
PYTHONPATH=src:third_party/model-evo-harness/src python -m ltvevo search \
  --task /home/charles/LTVEvo-data/retail-90d-task.json \
  --snapshots /home/charles/LTVEvo-data/snapshots/retail-90d \
  --journal /home/charles/LTVEvo-data/runs/retail-90d-harness-live/journal.json \
  --steps 1 --device cuda --harness model-evo \
  --provider-url https://api.deepseek.com --model deepseek-flash --thinking enabled
# Resume the same journal, increasing the total step count to two:
PYTHONPATH=src:third_party/model-evo-harness/src python -m ltvevo search \
  --task /home/charles/LTVEvo-data/retail-90d-task.json \
  --snapshots /home/charles/LTVEvo-data/snapshots/retail-90d \
  --journal /home/charles/LTVEvo-data/runs/retail-90d-harness-live/journal.json \
  --steps 2 --device cuda --harness model-evo \
  --provider-url https://api.deepseek.com --model deepseek-flash --thinking enabled
```

## Results

MAE is lower-is-better. Paired intervals are customer-cluster bootstrap 95% intervals for **reference MAE minus candidate MAE** on the same validation rows (1,000 draws; 4,537 customer clusters). A positive interval would favor the candidate.

| Validation candidate | MAE | Paired improvement vs all-zero [95% CI] | CUDA evidence |
| --- | ---: | ---: | --- |
| Historical-spend baseline | 423.29 | — | Reference |
| All-zero reference, retained champion | **319.26** | 0 | Reference |
| Agent step 1: pinball median + internal-holdout shrinkage | 319.38 | -0.12 [-1.06, 0.89] | `model_device=cuda`, `prediction_device=cuda:0`, 23,285,248 peak bytes |
| Agent step 2: propensity/positive-median ranking + top-k constant | 334.05 | -14.79 [-23.76, -5.80] | `model_device=cuda`, `prediction_device=cuda:0`, 23,657,984 peak bytes |

Step 1 was close to zero but its interval includes no clear gain. Step 2 is clearly worse on MAE by this paired interval. Both candidates were evaluated on **all 13,292 validation rows**; predictions were finite and nonzero for 13,292 rows in step 1 and 2,660 rows in step 2. The candidate source SHA-256 values are `e3a7a7926254217be55208705d712c2f1637d3976c1403c565a2ba2c63182e1d` and `3f404a32014ba8c7a6f23dcafbe8e5849843e52b5075e42b0410f8ecc80323f2`, respectively. Both reflections marked the hypothesis invalid and recorded `business_experience.status=not_observable`. The journal retained `zero_reference` as `best_id`.

The second hypothesis explicitly cited step 1's MAE and replaced its continuous-value shrinkage with ranking plus a holdout-selected top-k constant. This demonstrates that the Agent used measured feedback to change the next experiment. The hypotheses were validated by ModelEvoHarness's research contract; neither proposal requested bundled reference source in this live run (`reference_reads` was empty). The earlier [deterministic GPU integration check](model-evo-harness-gpu-2026-10-06.md) exercised bounded reference reading with a SHA-256 ledger.

## Candidate implementation audit and limits

Independent source review found no validation-label leakage: the host passes only validation features to candidate code. Both generated modules place training and prediction tensors on CUDA and return CUDA tensors. The review also found two implementation risks in the **generated candidates**:

1. Both modules silently replace a non-finite prediction vector with all zeros. This can conceal a model failure from the sandbox's finite-output check. The fallback was not taken in this run, as each output vector had nonzero finite predictions, but future candidates should fail explicitly on non-finite values.
2. Step 2 uses a score quantile and `>=` to select its top-k set. Tied scores can select more than k rows, so the implementation does not guarantee the exact top-k fraction stated in its hypothesis. Its observed validation output was nonzero for 2,660 of 13,292 rows.

The host records `implementation_status=unverified` and `attribution=unverified` in technical experience; Harness validation of research text does not prove candidate implementation correctness. These findings should be addressed before treating an Agent-generated candidate as a trustworthy model improvement. This run establishes the GPU Agent loop, measured rejection of two candidates, and preservation of the validation champion. It does not establish an LTV model-quality gain or a marketing treatment effect.
