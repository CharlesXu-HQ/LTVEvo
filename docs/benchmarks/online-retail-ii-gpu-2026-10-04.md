# Full-data Online Retail II GPU validation

Date: 2026-10-04. This is a record of an experiment run, including a negative generalization result. Raw data and API credentials are not stored in the repository.

## Data and frozen task

- Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii), downloaded on the experiment machine through the [full CSV mirror](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci). The downloaded ZIP contains `online_retail_II.csv` with 1,067,371 transaction rows.
- Raw ZIP SHA-256: `e2a0ebc53c3b0b577ff2f3dc4eb64032a38e38f6d7a47f73e511cdcd6d5b7dbd`.
- Task fingerprint: `44c020de3029426b3919ba869633c2c092d8a2de1cbbd74fde1cc62bcef93e61`. The matching manifest is [`examples/online-retail-ii-kaggle-90d.json`](../../examples/online-retail-ii-kaggle-90d.json).
- Target: known customers' positive purchase amount before refunds in the next 90 days. This is a fixed-horizon spending proxy, not net revenue, margin, or causal marketing lift.
- Observation dates: monthly; train 2010-03 through 2010-09, validation 2011-01 through 2011-03, test 2011-07 through 2011-09. Label windows end before the next split begins.

All 1,067,371 source rows were read. The adapter retained 805,549 eligible purchase rows, of which 786,124 occur before the final label-window end. Sequential exclusions: 243,007 missing customer IDs, 18,744 cancellation rows, 71 nonpositive-price rows, 0 invalid timestamps, and 0 additional nonpositive-quantity rows. Another 19,425 valid purchase rows fall after the last label window and were counted, not sampled away. Snapshot counts were 18,472 train, 13,292 validation, and 15,433 test.

| Split | Known customers | Zero-target rate | Mean 90-day target |
| --- | ---: | ---: | ---: |
| Train | 3,321 | 45.82% | 630.28 |
| Validation | 4,537 | 66.45% | 319.26 |
| Test | 5,249 | 61.84% | 447.94 |

Split CSV SHA-256: train `b613b20af0a261bc04979bf12e459c309218f36f152651294fc3117bd97b9609`; validation `ed6af45b54043558e0e189718c6525693436ad2e206718412ec99cc7be1f6550`; test `f5efbcccd40eacc9463c894d7f973327d27de4e26193ac744520fb53cb677cf3`.

## Execution evidence

Machine: NVIDIA GeForce RTX 5090, 32,607 MiB, driver 580.178.04. Host PyTorch 2.11.0+cu128; isolated candidate image `ltvevo-sandbox` built from PyTorch 2.11.0 CUDA 12.8. The container completed a CUDA tensor calculation before experiments. Every scored candidate returned a CUDA tensor (`prediction_device=cuda:0`) and recorded nonzero peak CUDA allocation. This demonstrates GPU prediction, not that every operation in Agent-written code was performed on GPU.

The provider was DeepSeek V4.1 Flash through model ID `deepseek-flash`; `high` and `max` requests were both checked against the live API with thinking enabled. Search proposals, reflections, and final analyses used `high`. The first run persisted every hypothesis, candidate source hash, metric, reflection, and final analysis in `/home/charles/LTVEvo-data/runs/retail-90d/journal.json`. The second run read three validation-only lessons from that finalized journal under the same task fingerprint and split hashes.

## Results

MAE is lower-is-better. `Top 10%` is the share of actual purchase amount captured by the customers with the highest predicted value. No ranking confidence interval was computed.

| Run / candidate | Validation MAE | Test MAE | Test top 10% |
| --- | ---: | ---: | ---: |
| Historical-spend baseline | 423.29 | 364.75 | 59.67% |
| PyTorch MLP seed | 746.45 | — | — |
| First run Agent step 1 | 412.02 | — | — |
| First run Agent step 2 | 385.27 | — | — |
| First run Agent step 3, selected | 338.31 | 369.83 | 62.66% |
| Second run all-zero reference, selected | 319.26 | 447.94 | 10.00% |
| Second run Agent step 1 | 464.53 | — | — |
| Second run Agent step 2 | 397.33 | — | — |

The first run omitted the all-zero reference during selection. Its Agent candidate reduced validation MAE from 423.29 to 338.31, but on the once-opened test split it was **5.08 worse** than the historical baseline. The paired customer-cluster 95% interval for MAE improvement was **[-33.43, 18.74]** (1,000 bootstrap draws; 5,249 customer clusters). The test does not confirm a primary-metric improvement. Its top-decile capture increased from 59.67% to 62.66%, but without a ranking interval this is descriptive.

The second run added the all-zero reference and correctly found that none of its evaluated candidates beat 319.26 validation MAE. The Agent stopped after two candidates; its statement that all-zero is the MAE *optimum* is stronger than the evidence. It was only the best **evaluated** validation candidate. The selected all-zero model had test MAE 447.94 versus 364.75 for the historical baseline. The paired difference was -83.19 with a 95% interval of [-127.31, -45.52]. This second test reused the same held-out rows **after the first run's test had already been inspected**, so it verifies workflow and exposes temporal drift; it is not a fresh, independent estimate for model development.

The second run's final journal stores a `high` report and a subsequent `max` anomaly review. The deterministic trigger was the validation-to-test MAE reversal and the negative paired interval against the historical baseline. The `max` review called a change in target level observable, but treated distribution shift and validation-selection overfitting as plausible explanations rather than proven causes; it found no evidence in the supplied aggregates that establishes leakage or target mismatch. This review did not reselect a model or rerun the held-out evaluation.

The validation period had many more zero targets and a much lower mean value than train or test. This explains why selecting by validation MAE favored a constant zero prediction and why that choice failed in the later period. It does not establish that the data preparation is wrong. For constant tied predictions, the evaluator intentionally assigns the same observed mean to each displayed decile and prorates top-decile capture; repeated decile means are expected, not evidence of label leakage.

## Reproduce

Download the full mirror ZIP directly to the experiment machine at `data/raw/online_retail_II_kaggle.zip`, verify its SHA-256 above, and use the exact manifest. The run used `https://www.kaggle.com/api/v1/datasets/download/mashlyn/online-retail-ii-uci` as the download URL. Build the CUDA sandbox on a machine with NVIDIA container support. Set `LTVEVO_API_KEY` in the environment without storing it in the repository.

```bash
ltvevo prepare --task examples/online-retail-ii-kaggle-90d.json --output data/snapshots/retail-90d
docker build -f Dockerfile.sandbox -t ltvevo-sandbox .
ltvevo baseline --task examples/online-retail-ii-kaggle-90d.json \
  --snapshots data/snapshots/retail-90d --journal runs/new-run/journal.json --device cuda
ltvevo search --task examples/online-retail-ii-kaggle-90d.json \
  --snapshots data/snapshots/retail-90d --journal runs/new-run/journal.json \
  --steps 5 --device cuda --provider-url https://api.deepseek.com --model deepseek-flash
ltvevo finalize --snapshots data/snapshots/retail-90d --journal runs/new-run/journal.json \
  --device cuda --provider-url https://api.deepseek.com --model deepseek-flash
```

The two completed run journals remain on the target machine at `/home/charles/LTVEvo-data/runs/retail-90d/` and `/home/charles/LTVEvo-data/runs/retail-90d-v2/`. A future generalization claim needs an independently held-out later period or another dataset and a predeclared selection rule that accounts for both amount error and ranking.
