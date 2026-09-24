"""Stage-local process-tree memory monitoring."""

from __future__ import annotations

import os
import threading
import time

import psutil


class PeakMemoryMonitor:
    def __init__(self, interval_seconds: float = 0.2):
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        self._sample()
        return self.peak_bytes / 1024**3

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        try:
            process = psutil.Process(os.getpid())
            processes = [process, *process.children(recursive=True)]
            total = 0
            for item in processes:
                try:
                    total += item.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak_bytes = max(self.peak_bytes, total)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

