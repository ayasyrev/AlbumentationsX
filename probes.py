"""Standalone NumKong correctness probes; no Albumentations, Albucore or Torch."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numkong as nk
import numpy as np


def digest(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def observe(value, expected, label, mode):
    before = digest(value)
    actual = tuple(int(item) for item in nk.moments(value))
    unchanged = digest(value) == before
    assert unchanged, label
    return {
        "label": label,
        "mode": mode,
        "dtype": str(value.dtype),
        "shape": value.shape,
        "strides": value.strides,
        "c_contiguous": value.flags.c_contiguous,
        "size": value.size,
        "input_sha256": before,
        "input_unchanged": unchanged,
        "expected": expected,
        "actual": actual,
        "correct": actual == tuple(expected),
        "sumsq_deficit": expected[1] - actual[1],
    }


def constants(mode):
    # Source-guided last passing lengths, followed by the first failing length.
    boundaries = {1: 528401, 2: 528416, 3: 363286, 4: 264208,
                  5: 231182, 6: 396312, 7: 330260, 8: 132104}
    manifest = set()
    for stride, last_good in boundaries.items():
        for count in (last_good - 1, last_good, last_good + 1, last_good + 2):
            manifest.add((count, stride, 255))
    for stride in (1, 4, 5, 8, 9, 16):
        for count in (0, 1, 31, 32, 33, 300000, 1000000, 2097152, 2097153):
            manifest.add((count, stride, 255))
    for stride in (1, 4, 5, 8):
        for fill in (0, 1, 127, 254):
            manifest.add((300000, stride, fill))
    for count, stride, fill in sorted(manifest):
        source = np.full(count * stride, fill, dtype=np.uint8)
        value = source[::stride]
        expected = (count * fill, count * fill * fill)  # Python integers, no dtype overflow.
        yield observe(value, expected, f"constant-{fill}-n{count}-stride{stride}", mode)
    for stride in (1, 3, 4, 5, 8, 9):
        value = np.full((600, 500, stride), 255, dtype=np.uint8)[..., 0]
        yield observe(value, (300000 * 255, 300000 * 65025), f"rank2-stride{stride}", mode)


def originals(mode):
    for channels in (4, 5):
        original = (np.random.default_rng(137).uniform(0.1, 0.9, (1024, 1024, channels)).astype(np.float32)
                    * 255).astype(np.uint8)
        expected = (int(original[..., 0].sum(dtype=np.uint64)),
                    int(np.square(original[..., 0].astype(np.uint64)).sum(dtype=np.uint64)))
        for layout, value in (
            ("interleaved", original[..., 0]),
            ("planar", original.transpose(2, 0, 1).copy().transpose(1, 2, 0)[..., 0]),
            ("flat", original[..., 0].copy().reshape(-1)),
        ):
            yield observe(value, expected, f"original-C{channels}-{layout}", mode)


def resource_preflight():
    def ticks():
        values = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        return sum(values), values[3] + values[4]
    first = ticks()
    time.sleep(0.25)
    second = ticks()
    busy = 1 - (second[1] - first[1]) / max(second[0] - first[0], 1)
    available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
    return {"load": os.getloadavg(), "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)), "cpu_busy_fraction_250ms": busy,
            "memory_available_kib": available_kib,
            "held_for_resource_load": busy > 0.80 or available_kib < 1024**2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--mode", choices=("automatic", "haswell", "serial"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initial = nk.get_capabilities()
    cpu_lines = Path("/proc/cpuinfo").read_text().splitlines()
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "site": args.site, "mode": args.mode,
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "os_release": Path("/etc/os-release").read_text(),
        "cpu": next(line.split(":", 1)[1].strip() for line in cpu_lines if line.startswith("model name")),
        "cpu_flags": next(line.split(":", 1)[1].strip() for line in cpu_lines if line.startswith("flags")),
        "numkong": importlib.metadata.version("numkong"), "numpy": np.__version__,
        "numkong_module": nk.__file__, "numpy_module": np.__file__,
        "capabilities_initial": initial,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_modules": {
            name: {"path": module.__file__, "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
            for name, module in sys.modules.items()
            if name.startswith("numkong") and getattr(module, "__file__", "").endswith(".so")
        }, "resource_preflight": resource_preflight(), "cases": [],
    }
    if report["resource_preflight"]["held_for_resource_load"]:
        report.update(status="held_for_resource_load", process_exit=3)
    elif args.mode == "haswell" and not initial.get("haswell"):
        report.update(status="unsupported_haswell_skipped", process_exit=0)
    else:
        keep = set(initial) if args.mode == "automatic" else {"serial", args.mode}
        for name, enabled in initial.items():
            if enabled and name not in keep:
                nk.disable_capability(name)
        effective = nk.get_capabilities()
        assert all(not enabled or initial[name] for name, enabled in effective.items())
        report["capabilities_effective"] = effective
        # Source inference, not a runtime function-pointer trace. The x86 uint8
        # moments order is identical in the 7.8.4 and 7.8.5 dispatch_u8.c files.
        report["backend_from_x86_dispatch_source"] = next(
            (name for name in ("icelake", "skylake", "sierra", "alder", "haswell", "serial") if effective.get(name)),
            "unknown",
        )
        report["cases"] = list(constants(args.mode)) + list(originals(args.mode))
        wrong = sum(not row["correct"] for row in report["cases"])
        report.update(status="complete", process_exit=int(bool(wrong)),
                      summary={"cases": len(report["cases"]), "wrong_results": wrong,
                               "all_inputs_unchanged": all(row["input_unchanged"] for row in report["cases"])})
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("site", "numkong", "mode", "status", "process_exit")} |
                     {"summary": report.get("summary"), "effective": report.get("capabilities_effective")}), flush=True)
    return report["process_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
