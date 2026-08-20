from __future__ import annotations

import json
import os
import platform
import resource
import shutil
import subprocess
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .device import optional_module, resolve_torch_device


@dataclass
class MemorySnapshot:
    """CPU/GPU memory snapshot used in runtime profiling reports."""

    rss_mb: float
    max_rss_mb: float
    python_traced_current_mb: float | None = None
    python_traced_peak_mb: float | None = None
    gpu_allocated_mb: float | None = None
    gpu_reserved_mb: float | None = None
    gpu_name: str | None = None
    gpu_total_mb: float | None = None
    gpu_used_mb: float | None = None


@dataclass
class ProfileReport:
    """Computational-cost report for one model or pipeline step."""

    name: str
    wall_time_s: float
    cpu_time_s: float
    start: MemorySnapshot
    end: MemorySnapshot
    delta_rss_mb: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a serializable dictionary."""
        data = asdict(self)
        data["throughput_items_per_s"] = None
        n_items = self.metadata.get("n_items")
        if n_items and self.wall_time_s > 0:
            data["throughput_items_per_s"] = n_items / self.wall_time_s
        return data


class Profiler:
    """Small dependency-light profiler for CPU RAM, GPU RAM and elapsed time."""

    def __init__(self, enabled: bool = True, device: str = "auto") -> None:
        self.enabled = enabled
        self.device = resolve_torch_device(device)
        self.reports: list[ProfileReport] = []
        if enabled and not tracemalloc.is_tracing():
            tracemalloc.start()

    def snapshot(self) -> MemorySnapshot:
        """Capture current process and optional GPU memory usage."""
        current_peak = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (None, None)
        gpu = self._gpu_snapshot()
        return MemorySnapshot(
            rss_mb=_current_rss_mb(),
            max_rss_mb=_max_rss_mb(),
            python_traced_current_mb=_bytes_to_mb(current_peak[0]),
            python_traced_peak_mb=_bytes_to_mb(current_peak[1]),
            **gpu,
        )

    @contextmanager
    def track(self, name: str, **metadata: Any) -> Iterator[None]:
        """Measure elapsed time and memory for a named operation."""
        if not self.enabled:
            yield
            return
        start_snapshot = self.snapshot()
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        yield
        end_cpu = time.process_time()
        end_wall = time.perf_counter()
        end_snapshot = self.snapshot()
        self.reports.append(
            ProfileReport(
                name=name,
                wall_time_s=end_wall - start_wall,
                cpu_time_s=end_cpu - start_cpu,
                start=start_snapshot,
                end=end_snapshot,
                delta_rss_mb=end_snapshot.rss_mb - start_snapshot.rss_mb,
                metadata=metadata,
            )
        )

    def final_report(self) -> dict[str, Any]:
        """Return the complete profiling report."""
        return {
            "system": system_info(self.device),
            "reports": [report.to_dict() for report in self.reports],
        }

    def print_final_report(self) -> None:
        """Print the formatted profiling report."""
        print(format_profile_report(self.final_report()))

    def save_json(self, path: str | Path) -> None:
        """Write the profiling report as JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.final_report(), indent=2, ensure_ascii=False), encoding="utf-8")

    def _gpu_snapshot(self) -> dict[str, float | str | None]:
        torch = optional_module("torch")
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            index = torch.cuda.current_device() if self.device == "cuda" else int(self.device.split(":", 1)[1])
            props = torch.cuda.get_device_properties(index)
            return {
                "gpu_allocated_mb": _bytes_to_mb(torch.cuda.memory_allocated(index)),
                "gpu_reserved_mb": _bytes_to_mb(torch.cuda.memory_reserved(index)),
                "gpu_name": props.name,
                "gpu_total_mb": _bytes_to_mb(props.total_memory),
                "gpu_used_mb": _nvidia_smi_used_mb(index),
            }
        return {
            "gpu_allocated_mb": None,
            "gpu_reserved_mb": None,
            "gpu_name": None,
            "gpu_total_mb": None,
            "gpu_used_mb": None,
        }


def system_info(device: str = "auto") -> dict[str, Any]:
    """Return startup information for CPU/GPU availability and memory."""
    resolved = resolve_torch_device(device)
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "device": resolved,
        "initial_rss_mb": _current_rss_mb(),
        "initial_max_rss_mb": _max_rss_mb(),
        "gpus": [],
    }
    torch = optional_module("torch")
    if torch is not None and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["gpus"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_mb": _bytes_to_mb(props.total_memory),
                    "used_mb": _nvidia_smi_used_mb(index),
                }
            )
    return info


def format_profile_report(report: dict[str, Any]) -> str:
    """Format a profile report as a readable final log."""
    lines = ["\n=== Entity Linking profiling report ==="]
    system = report["system"]
    lines.append(
        f"System: device={system['device']} cpu_count={system['cpu_count']} "
        f"initial_rss_mb={system['initial_rss_mb']:.2f}"
    )
    if system["gpus"]:
        for gpu in system["gpus"]:
            lines.append(f"GPU[{gpu['index']}]: {gpu['name']} total_mb={gpu['total_mb']:.2f} used_mb={gpu['used_mb']}")
    else:
        lines.append("GPU: not available or not detected")
    for item in report["reports"]:
        throughput = item.get("throughput_items_per_s")
        throughput_text = f" throughput={throughput:.2f} items/s" if throughput is not None else ""
        lines.append(
            f"[{item['name']}] wall={item['wall_time_s']:.4f}s cpu={item['cpu_time_s']:.4f}s "
            f"rss_delta={item['delta_rss_mb']:.2f}MB end_rss={item['end']['rss_mb']:.2f}MB"
            f" gpu_alloc={item['end']['gpu_allocated_mb']}MB gpu_reserved={item['end']['gpu_reserved_mb']}MB"
            f"{throughput_text} metadata={item['metadata']}"
        )
    return "\n".join(lines)


def _current_rss_mb() -> float:
    # Linux /proc is more accurate for current RSS than ru_maxrss.
    statm = Path("/proc/self/statm")
    if statm.exists():
        rss_pages = int(statm.read_text().split()[1])
        return rss_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    return _max_rss_mb()


def _max_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / (1024 * 1024)
    return value / 1024


def _bytes_to_mb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024 * 1024)


def _nvidia_smi_used_mb(index: int) -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return float(result.stdout.strip().splitlines()[0])
