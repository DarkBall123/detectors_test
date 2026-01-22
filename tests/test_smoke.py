import numpy as np

from app.benchmark import Benchmark
from app.draw import draw_detections
from app.types import Detection


def test_draw_detections_smoke():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    draw_detections(frame, [Detection(5, 5, 20, 20, 0.9)])
    assert frame.sum() > 0


def test_benchmark_summary_smoke():
    bench = Benchmark(warmup=0)
    bench.update(0.01)
    bench.update(0.02)
    summary = bench.summary()
    assert summary["frames"] == 2
    assert summary["avg_fps"] > 0
