# NumKong uint8 moments overflow evidence

The issue reproducer and full 114-case correctness matrix use only NumPy and
NumKong. Inputs are synthetic. No performance measurements are included.

Download [the complete evidence archive](numkong-moments-evidence-20261021.zip).
The short MRE and full runner are also available as files in this branch.

Archive SHA-256: `e993f153ab3269d0c3abf190749a2a049e81ba15ec7600b63a5bc88394eec000`.

## Reproduce

Use Python 3.13.14 and NumPy 2.5.1. Run both NumKong 7.8.4 and 7.8.5 in separate
environments. For example, with uv:

```bash
uv venv --python 3.13.14 .venv-7.8.5
uv pip install --python .venv-7.8.5/bin/python numpy==2.5.1 numkong==7.8.5
uv run --no-project --python .venv-7.8.5/bin/python python reproduce.py
uv run --no-project --python .venv-7.8.5/bin/python python probes.py --site reproduction --mode automatic --output automatic.json
uv run --no-project --python .venv-7.8.5/bin/python python probes.py --site reproduction --mode haswell --output haswell.json
uv run --no-project --python .venv-7.8.5/bin/python python probes.py --site reproduction --mode serial --output serial.json
```

Each mode runs in a fresh process. Haswell mode disables alternative detected
capabilities; it never enables an unsupported instruction set. If Haswell is not
available, the full probe records a skip. Serial disables non-serial capabilities.
Automatic keeps the initial mask. The minimal reproducer prints capabilities and
isolates Haswell before comparing with serial.

Probe exit 1 means at least one incorrect numerical result was captured. Exit 0
means all numerical results pass, or an explicitly recorded unsupported mode was
skipped. Exit 3 means resource load held execution. Check JSON `status` and
`summary`; an execution or setup failure is not numerical evidence.

## Archive contents and interpretation

- `probes.py`, `reproduce.py`: complete executable reproducers.
- `results/{local,gcp,kaggle}/`: every observation from both versions and all
  three modes, including expected/actual results, shapes, strides, input hashes,
  mutation checks, CPU/OS metadata, effective capabilities, native-module hashes
  and resource preflight. Machine-local filesystem paths have been omitted;
  numerical observations are unchanged from the captured original JSON files.
- `summary.json`: compact per-process counts and input-equivalence checks.
- `SHA256SUMS.json`: hashes of the files in this archive.

Counts summarize this fixed matrix, not all uint8 inputs. Values and lengths
around the observed overflow boundaries are deliberately represented heavily.
The inferred backend label follows the x86 dispatcher source order; it is not a
runtime function-pointer trace. Full capability masks are retained.

Source reference:
[NumKong v7.8.5 Haswell moments](https://github.com/ashvardanian/NumKong/blob/337c1cb6e6fbea8c08b3b7b385e392ed95ccab46/include/numkong/reduce/haswell.h#L1141-L1231).
