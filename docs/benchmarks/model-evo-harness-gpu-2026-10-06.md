# Pinned ModelEvoHarness integration: full-data GPU check

Date: 2026-10-06. This is an **integration check**, not evidence that an autonomous Agent discovered a better LTV model. The Agent responses were deterministic fixtures; no live model API or API key was used in this run.

## Frozen data and code

- Source: complete Online Retail II mirror ZIP already prepared on the GPU host. All **1,067,371 raw transaction rows** were read during preparation, with no sampling. The raw ZIP SHA-256 is `e2a0ebc53c3b0b577ff2f3dc4eb64032a38e38f6d7a47f73e511cdcd6d5b7dbd`; eligibility and split accounting are in the [original full-data record](online-retail-ii-gpu-2026-10-04.md).
- Frozen task fingerprint: `44c020de3029426b3919ba869633c2c092d8a2de1cbbd74fde1cc62bcef93e61`. The existing snapshots contain **18,472 train**, **13,292 validation**, and **15,433 test** customer-observation rows. No split was re-sampled or modified.
- LTVEvo code commit: `fe0d5c416d8f771361f87f2596872b35f780e598`; pinned ModelEvoHarness commit: `23c947ddc90b0a6ebafddfebad228021f1a54d5e`. The submodule had no local changes. ModelEvoHarness itself was not modified.
- Device: NVIDIA GeForce RTX 5090. Host Python 3.12.13, PyTorch 2.11.0+cu128, CUDA available; candidates ran in the existing `ltvevo-sandbox` GPU image.

## Procedure

The `baseline` command used `--harness model-evo --device cuda` on the existing complete snapshots. A fresh journal then ran one full Agent-loop smoke experiment in the same mode. A deterministic provider fixture first requested Harness's bundled PyTorch training reference, then proposed the existing PyTorch MLP with its MSE loss changed to smooth L1. The proposal declared the local `ltv_tabular_regression` family, same-validation comparison, expected MAE improvement, falsification condition, actual input fields, and an alternative. Harness validated the research, returned a SHA-256 reference ledger, and validated the technical/business reflection. The reflection explicitly recorded `business_experience.status=not_observable`.

The candidate code SHA-256 is `41371f7392a0dd7cc2f257efbed1279de35acf0b7b9d934ce735a9026638a337`. Harness read `models/pytorch/training.py` with SHA-256 `22cd259ab713f290491be36b363e253d63acf91b39c7739f4157db6171b3acc8`. The fixture made two `high` proposal calls (reference request, then experiment) and one `high` reflection call. It made no network model call. The candidate trained and predicted on CUDA: `model_device=cuda`, `prediction_device=cuda:0`, peak CUDA allocation **26,711,552 bytes**.

## Measured result

| Candidate | Validation MAE | Test MAE |
| --- | ---: | ---: |
| Historical-spend baseline | 423.29 | 364.75 |
| All-zero reference, selected | **319.26** | 447.94 |
| PyTorch MLP seed | 746.45 | — |
| Smooth L1 smoke candidate | 1,499.61 | — |

Lower MAE is better. The robust-loss candidate failed its hypothesis by a wide margin, so the journal kept `zero_reference` as the best **evaluated** validation candidate. Finalization evaluated that frozen choice once on the test rows. Against the historical-spend baseline, its test MAE improvement was **-83.19**, with a paired customer-cluster 95% interval **[-127.31, -45.52]** (1,000 bootstrap draws, 5,249 customer clusters). This reproduces the known weakness of selecting solely on this validation period's MAE; it is not a new model-quality gain.

The same public test rows were inspected in the earlier benchmark. This finalization checks the one-time execution path of this new journal and is **not an independent confirmatory holdout**. A live LLM Agent experiment under the new Harness mode remains unverified; the previous live DeepSeek experiment predates this integration. The complete integration journal is on the target machine at `/home/charles/LTVEvo-data/runs/retail-90d-harness-smoke-v2/journal.json`.
