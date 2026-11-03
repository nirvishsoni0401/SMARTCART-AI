import time
from collections import deque
from typing import Dict, Any, List
import numpy as np


class LatencyTracker:
    """Thread-safe in-memory sliding window latency monitor."""

    def __init__(self, maxlen: int = 2000):
        self.maxlen = maxlen
        self.e2e_latencies: deque = deque(maxlen=maxlen)
        self.model_scoring_latencies: deque = deque(maxlen=maxlen)
        self.endpoint_latencies: Dict[str, deque] = {}

    def record_e2e(self, path: str, duration_ms: float) -> None:
        self.e2e_latencies.append(duration_ms)
        if path not in self.endpoint_latencies:
            self.endpoint_latencies[path] = deque(maxlen=500)
        self.endpoint_latencies[path].append(duration_ms)

    def record_scoring(self, duration_ms: float) -> None:
        self.model_scoring_latencies.append(duration_ms)

    def get_percentiles(self) -> Dict[str, Any]:
        def calc_pct(arr: List[float]):
            if not arr:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}
            np_arr = np.array(arr)
            return {
                "p50": round(float(np.percentile(np_arr, 50)), 2),
                "p95": round(float(np.percentile(np_arr, 95)), 2),
                "p99": round(float(np.percentile(np_arr, 99)), 2),
                "count": len(arr),
            }

        return {
            "end_to_end": calc_pct(list(self.e2e_latencies)),
            "model_scoring": calc_pct(list(self.model_scoring_latencies)),
        }


# Global singleton
latency_tracker = LatencyTracker()

