"""Run in a fresh process: isolate the detected Haswell backend, then serial."""

from importlib.metadata import version

import numkong as nk
import numpy as np

capabilities = nk.get_capabilities()
print("NumKong", version("numkong"), "NumPy", np.__version__)
print("Detected:", [name for name, enabled in capabilities.items() if enabled])
if not capabilities.get("haswell"):
    raise SystemExit("Haswell is unavailable on this host; this reproduction is skipped.")

# Disable alternatives; never enable a capability that was not detected.
for name, enabled in capabilities.items():
    if enabled and name not in {"serial", "haswell"}:
        nk.disable_capability(name)

samples = {
    "strided": np.full(1_200_000, 255, dtype=np.uint8)[::4],
    "contiguous": np.full(1_000_000, 255, dtype=np.uint8),
}
for backend in ("haswell", "serial"):
    if backend == "serial":
        nk.disable_capability("haswell")
    print("Enabled:", [name for name, enabled in nk.get_capabilities().items() if enabled])
    for name, values in samples.items():
        expected = (values.size * 255, values.size * 255**2)
        actual = nk.moments(values)
        print(backend, name, "actual:", actual, "expected:", expected)
        if backend == "serial":
            assert actual == expected
