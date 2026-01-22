from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class Benchmark:
    warmup: int = 30
    _times: List[float] = field(default_factory=list)
    _seen: int = 0

    def update(self, elapsed_s: float) -> None:
        self._seen += 1
        if self._seen <= self.warmup:
            return
        self._times.append(elapsed_s)

    def summary(self) -> Dict[str, float]:
        if not self._times:
            return {
                "frames": 0,
                "avg_fps": 0.0,
                "avg_ms": 0.0,
                "p95_ms": 0.0,
            }
        times = np.array(self._times, dtype=np.float32)
        avg_s = float(times.mean())
        avg_ms = avg_s * 1000.0
        p95_ms = float(np.percentile(times, 95)) * 1000.0
        avg_fps = 1.0 / avg_s if avg_s > 0 else 0.0
        return {
            "frames": len(self._times),
            "avg_fps": avg_fps,
            "avg_ms": avg_ms,
            "p95_ms": p95_ms,
        }
