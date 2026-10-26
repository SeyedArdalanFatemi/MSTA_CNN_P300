# Validation record — 2026-10-24

The release assembly was checked without TensorFlow installed locally.
It does not claim new training, new Test inference, or independent full reproduction.

Passed:
- Python syntax parsing for all included .py files.
- Core data/decoder self-test; ablation aggregation self-test; feature-analysis
  self-test; supplementary metrics self-test via run.py self-test.
- Core msta.py byte identity against the previously tested release:
  SHA256 1432f8727ae90f15ceaf45665406b69a921175dc000cd00d8e4246f81ee8b1e1.
- All 24 included .keras checkpoints byte-identical to previous reference bundles.
- Supplementary regeneration: all 10 CSVs byte-identical to archived results.
- Asset regeneration: all 7 CSVs byte-identical; 4 figures produced in 3 formats.
  Original figures had previously undergone visual inspection; newly regenerated
  files were checked for successful production, not a fresh detailed visual audit.
- New --evaluation-dir interface tested with an evaluation-format fixture built
  from archived predictions/reports: all CSVs identical. This was not new inference.
- Release manifest verification including reference-model bundle consistency.

Previously supplied user runs include TensorFlow smoke tests, reference evaluation
and full primary workflow summaries. This integration does not retrospectively
increase the evidence available for those runs. Original raw CV checkpoints are
not all included; regenerating features requires them or retraining.

Not executed in this environment:
- TensorFlow training/inference and the newly added transfer-recipe refit wrapper.
- Transfer refitting was not run locally; it was subsequently executed on the origin machine (see below).
- Full training on another computer; no promise of cross-platform bit identity.

The retained core still supports historical d50 ablation for compatibility.
The reported d25 matched experiment is run through run.py ablation.
No Test-dependent search or architecture change was added by integration.

## Origin-machine checks reported on 2026-10-24

The operator supplied successful logs for release verification, core data tests,
TensorFlow smoke tests (four variants at dropout0.25 and0.5), all unified non-TF
self-tests, supplementary analysis and asset generation. These are operator-reported
checks on the original Windows CPU environment, not a new independent replication.

For version2.0.0 the feature extractor retains the original tensor call used in
the completed feature experiment; the unexecuted list-input modification was reverted.
The known Keras input-structure warning can remain. No new feature results are claimed.
The main core and 24 reference checkpoints are unchanged. Public-facing documentation
was consolidated. Historical experiment metadata is retained for provenance.

## Version 2.0.1
MIT licensing and a sequential transfer-check launcher were added. The launcher
comparison is tested with synthetic records, not real training. TensorFlow and
the original recordings are unavailable in the assembly environment. At version 2.0.1 packaging time, the origin-machine run was pending. It has since
completed, as documented below.

## Version 2.0.2: transfer-run report audit

Evidence: evidence/SEND_transfer_check.zip (supplied by the operator).
The archive contains the transfer comparison, evaluation summary, final training
manifest and frozen protocol. All 12 training jobs and evaluations are complete.
The protocol declares no Test use for training. Per-job epoch counts, recorded
learning-rate schedules, finite training losses and optimizer update counts agree
with the frozen tasks. Protocol/manifest/evaluation/reference hash links were checked.
All 180 character-accuracy curve points (12 jobs x 15 repetitions) equal the archived
reference. Reported maximum save/reload probe difference is 0.0. Summed recorded
training time is 2442.4114792997716 seconds; this excludes other workflow overhead.

The submitted archive does not contain newly trained weight files or event-level
prediction arrays. Their bytes were not independently inspected in this audit.
Matching curve counts do not prove identical weights or identical event probabilities.
This is reproduction on the original machine, not independent cross-platform validation,
and not a fresh confirmatory test. The distributed checkpoints remain the earlier
reference checkpoints. No numerical result, selection rule or training algorithm
was changed in version 2.0.2. Only documentation, CLI help and evidence were updated.
