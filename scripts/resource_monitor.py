"""Progress bars and live local-resource statistics for S2SR runs.

Lightweight, dependency-free helpers used by the local engine:

- ``Bar`` renders a single-line progress bar (auto-throttled when stdout
  is redirected, e.g. inside mosaic tile logs).
- ``ResourceMonitor`` samples CPU, RAM, and CUDA memory in a background
  thread and reports both live snapshots and peaks for the run summary.
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from pathlib import Path


def _read_cpu_total() -> int:
    try:
        with open("/proc/stat", "rb") as handle:
            fields = handle.readline().split()[1:]
        return sum(int(value) for value in fields)
    except (OSError, ValueError):
        return 0


def _read_cpu_idle() -> int:
    try:
        with open("/proc/stat", "rb") as handle:
            idle = int(handle.readline().split()[4])
        return idle
    except (OSError, ValueError):
        return 0


def _read_ram() -> tuple[float, float]:
    """Return (used_gb, total_gb) from /proc/meminfo."""
    try:
        values = {}
        with open("/proc/meminfo", "rb") as handle:
            for line in handle:
                key, value = line.decode().split(":", 1)
                if key in ("MemTotal", "MemAvailable"):
                    values[key] = int(value.strip().split()[0]) / 1024**2
        total = values.get("MemTotal", 0.0)
        used = total - values.get("MemAvailable", 0.0)
        return used, total
    except (OSError, ValueError):
        return 0.0, 0.0


def _cuda_memory() -> tuple[float, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return (
            (total_bytes - free_bytes) / 1024**3,
            total_bytes / 1024**3,
        )
    except Exception:
        return None


class Bar:
    """Single-line progress bar; throttles updates when output is not a TTY."""

    def __init__(self, total: int, label: str, width: int = 28) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.width = width
        self.current = 0
        self._is_tty = sys.stdout.isatty()
        self._last_render = 0.0
        self._render()

    def update(self, amount: int = 1) -> None:
        self.current = min(self.current + amount, self.total)
        now = time.monotonic()
        if self._is_tty or self.current == self.total or now - self._last_render >= 15:
            self._last_render = now
            self._render()

    def _render(self) -> None:
        ratio = self.current / self.total
        filled = int(ratio * self.width)
        bar = "#" * filled + "-" * (self.width - filled)
        suffix = "" if self._is_tty else f" [{time.strftime('%H:%M:%S')}]"
        end = "\n" if self.current >= self.total else ""
        print(
            f"\r  {self.label} [{bar}] {self.current}/{self.total}{suffix}",
            end=end,
            flush=True,
        )


class ResourceMonitor:
    """Background sampler of CPU/RAM/CUDA-memory usage with peak tracking."""

    def __init__(self, interval: float = 2.0, echo_seconds: float = 30.0) -> None:
        self.interval = interval
        self.echo_seconds = echo_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.started_at: float | None = None
        self.peaks = {
            "cpu_percent": 0.0,
            "ram_used_gb": 0.0,
            "vram_used_gb": 0.0,
        }
        self.snapshot = ""

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    def start(self) -> None:
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)

    def _loop(self) -> None:
        last_total, last_idle = _read_cpu_total(), _read_cpu_idle()
        last_echo = time.monotonic()
        while not self._stop.is_set():
            time.sleep(self.interval)
            total, idle = _read_cpu_total(), _read_cpu_idle()
            busy_ticks = max(total - last_total, 1)
            cpu = 100.0 * (1.0 - (idle - last_idle) / busy_ticks)
            last_total, last_idle = total, idle
            ram_used, ram_total = _read_ram()
            cuda = _cuda_memory()
            parts = [f"CPU {min(max(cpu, 0.0), 100):5.1f}%", f"RAM {ram_used:.1f}/{ram_total:.1f}GB"]
            if cuda is not None:
                parts.append(f"VRAM {cuda[0]:.1f}/{cuda[1]:.1f}GB")
            with self._lock:
                self.peaks["cpu_percent"] = max(self.peaks["cpu_percent"], cpu)
                self.peaks["ram_used_gb"] = max(self.peaks["ram_used_gb"], ram_used)
                if cuda is not None:
                    self.peaks["vram_used_gb"] = max(self.peaks["vram_used_gb"], cuda[0])
                self.snapshot = " | ".join(parts)
            now = time.monotonic()
            if self.echo_seconds and now - last_echo >= self.echo_seconds:
                last_echo = now
                elapsed = int(now - (self.started_at or now))
                print(f"  [{elapsed // 60:02d}:{elapsed % 60:02d}] {self.snapshot}", flush=True)

    def summary(self) -> list[str]:
        elapsed = (
            time.monotonic() - self.started_at if self.started_at else 0.0
        )
        minutes, seconds = divmod(int(elapsed), 60)
        lines = [
            f"resources: {', '.join(self.snapshot.split(' | '))}"
            if self.snapshot
            else "resources: no samples",
            f"peaks: CPU {self.peaks['cpu_percent']:.1f}%"
            f", RAM {self.peaks['ram_used_gb']:.1f} GB",
        ]
        if self.peaks["vram_used_gb"]:
            lines[-1] += f", VRAM {self.peaks['vram_used_gb']:.1f} GB"
        lines[-1] += f" | wall time {minutes}m{seconds:02d}s"
        disk_free = shutil.disk_usage(Path.cwd().anchor or "/").free / 1024**3
        lines.append(f"disk free now: {disk_free:.0f} GB")
        return lines
