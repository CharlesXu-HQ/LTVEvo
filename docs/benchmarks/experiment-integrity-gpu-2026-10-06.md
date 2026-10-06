# Full-data GPU regression for experiment integrity

Date: 2026-10-06. This is a deterministic workflow check of LTVEvo commit `5e65112`, with ModelEvoHarness pinned at `23c947d`. It does **not** use a live Agent API or claim a better LTV model.

The run used the existing complete Online Retail II snapshots described in the [full-data preparation record](online-retail-ii-gpu-2026-10-04.md): raw ZIP SHA-256 `e2a0ebc53c3b0b577ff2f3dc4eb64032a38e38f6d7a47f73e511cdcd6d5b7dbd`, 18,472 training rows and all 13,292 validation rows. The target machine was an NVIDIA RTX 5090. Search did not call `finalize` or open the held-out test labels.

The deterministic proposal callback supplied two candidates through the normal `run_search(..., device="cuda", harness="model-evo")` host loop. The first returned a CUDA constant-one tensor to exercise evaluation; the second contained a non-finite-to-zero fallback to exercise the source audit. The callback also supplied fixed reflections, so this check does not validate Agent reasoning or Harness research-text validation. The real Agent/Harness loop was tested separately in the [live GPU run](model-evo-harness-live-gpu-2026-10-06.md).

| Result | Evidence |
| --- | --- |
| Full validation evaluated | 13,292 rows, 4,537 customer clusters, three observation periods |
| PyTorch seed | `prediction_device=cuda:0`, peak CUDA allocation 26,711,552 bytes |
| Constant-one candidate | `prediction_device=cuda:0`, MAE 319.5922 versus all-zero 319.2631; paired improvement -0.3291, 95% interval [-0.3560, -0.3044] |
| Source contradiction | Second candidate marked `failed` before GPU execution; journal records the non-finite-to-zero finding |
| Selection | `zero_reference` retained; first candidate had no point gain, second was blocked by source audit |
| Artifacts | Validation predictions for evaluated entries carry SHA-256 hashes; the journal has no `final` field |

The pinned checkout passed 80 tests on the target machine. This check establishes that the new audit, period diagnostics, artifact hashes, and selection record work with the full frozen dataset and CUDA runner. It does not establish statistical generalization, a model-quality gain, or a marketing treatment effect.
