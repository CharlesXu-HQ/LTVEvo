# ModelEvoHarness integration plan

Spec: `docs/specs/2026-10-06-model-evo-harness-alignment.md`.

1. Pin ModelEvoHarness as a submodule and add the LTV task/catalog adapter. Verify a real package import, honest `ltv_prediction` applicability, and stable identity hashes.
2. Add `--harness model-evo` to Agent search. Feed a frozen training/validation context and host-owned reference hashes to the Agent, use Harness source-reading and research validation, then persist research in the existing journal. Verify invalid or cross-stage proposals cannot reach the candidate runner.
3. Validate structured reflection with technical and non-observable business experience. Preserve existing exact-task experience, `high`/`max` review, resume, GPU checks, and final test seal. Verify these behaviors through integration tests.
4. Update both READMEs, CLI help, install instructions, and CI. Run the full test suite, build the package, and execute an end-to-end full-data run on the target GPU. Record any model-quality limitations separately from integration status.
