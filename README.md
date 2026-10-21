# Issue #584: benchmark evidence

Evidence for [AlbumentationsX issue #584](https://github.com/albumentations-team/AlbumentationsX/issues/584).
This branch stores review artifacts separately from the fix's source branch.

- [Complete evidence archive](benchmark-evidence.zip): raw timings, full tables,
  scripts, source snapshots, validation records and reproduction instructions.
- [Original baseline-versus-fix comparison: 120 cells](baseline-vs-fix.csv).
- [Expanded baseline/C/K comparison: 671 cells](baseline-c-vs-k.csv).

The matrices are separate runs, not 791 independent workloads. The PR's compact
table uses the expanded CSV. The archive's `pr-summary-cells.json` identifies
the exact rows. Extract the ZIP and open its root `README.md` to navigate the
appendix and reproduction instructions. The archive retains the source reports
as historical records; their local-publication status statements predate this
upload.

Baseline: `45a1afe21448fd2e83716733e2880c00335060fb`.
Fix: `ff6db7b86e9094112a2e5a0ba2268ab4f199791f`.
The optional K-order variant was evaluated separately and is not part of the fix.

Times are medians per variant; percentage changes are medians of paired ratios.
The original baseline is unsafe for readonly storage. Two normalization cases
were excluded before timing after failing semantic validation; their observations
and subsequent independent checks remain available in the archive. The detailed
dependency investigation is kept in separate report and evidence files.

Measurements were made on Linux/Ryzen 7 5800X, Python 3.13.14, NumPy 2.5.1,
Torch 2.14.1+cpu and Albucore 0.2.21. They do not constitute cross-platform,
GPU or DataLoader performance guarantees.

Archive size: 1,192,005 bytes, 77 members. SHA-256:
`358fa4034aac01838681662ace8611518e3ee374ccec5c41764e143120546a29`.
The root `SHA256SUMS.json` records every other archive member's hash.
