"""
StreamMonitor — мониторинг потока данных на структурные изменения.

Разбивает входной поток на окна и отслеживает динамику w(t).
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .core import byte_stats


@dataclass
class Window:
    index: int
    start: int
    end: int
    frag: float
    H: float
    n_unique: int


class StreamMonitor:
    """
    Мониторит поток байтов, разбивая его на перекрывающиеся окна.

    Использование:
        monitor = StreamMonitor(window=4096, stride=2048)
        for chunk in stream:
            monitor.push(chunk)
        changes = monitor.detect_changes(threshold=2.0)
    """

    def __init__(self, window: int = 4096, stride: int = 2048):
        self.window = window
        self.stride = stride
        self._buffer = bytearray()
        self._windows: list[Window] = []
        self._next_start = 0

    def push(self, chunk: bytes):
        self._buffer.extend(chunk)
        while self._next_start + self.window <= len(self._buffer):
            start = self._next_start
            end = start + self.window
            data = bytes(self._buffer[start:end])
            st = byte_stats(data)
            self._windows.append(Window(
                index=len(self._windows),
                start=start, end=end,
                frag=st["frag"],
                H=st["H"],
                n_unique=st["n_unique"],
            ))
            self._next_start += self.stride

    @property
    def windows(self) -> list[Window]:
        return self._windows

    def detect_changes(self, metric: str = "frag",
                       z_threshold: float = 2.5) -> list[dict]:
        """
        Возвращает список окон, где метрика резко отличается от соседей.
        """
        if len(self._windows) < 5:
            return []
        vals = np.array([getattr(w, metric) for w in self._windows])
        # Локальное скользящее среднее по 5 окнам
        k = 5
        padded = np.pad(vals, (k // 2, k // 2), mode="edge")
        smooth = np.convolve(padded, np.ones(k) / k, mode="valid")
        resid = vals - smooth
        sigma = max(np.std(resid), 1e-9)
        z = resid / sigma
        changes = []
        for i, wz in enumerate(z):
            if abs(wz) > z_threshold:
                changes.append({
                    "window": i,
                    "start": self._windows[i].start,
                    "metric": metric,
                    "value": float(vals[i]),
                    "z": float(wz),
                })
        return changes

    def reset(self):
        self._buffer.clear()
        self._windows.clear()
        self._next_start = 0