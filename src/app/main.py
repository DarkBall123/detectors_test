from __future__ import annotations

import argparse
import json
import time
from typing import Optional

from tqdm import tqdm

from app.benchmark import Benchmark
from app.draw import draw_detections
from app.detectors.rtdetr_ultralytics import RTDETRDetector
from app.detectors.yolov8_ultralytics import YOLOv8Detector
from app.video import iter_frames, make_writer, open_video


def resolve_device(device: str) -> str:
    if device != "mps":
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    if not torch.backends.mps.is_available():
        return "cpu"
    return "mps"


def build_detector(model_name: str, device: str, conf: float):
    if model_name == "rtdetr":
        return RTDETRDetector(device=device, conf_thres=conf)
    if model_name == "yolov8":
        return YOLOv8Detector(device=device, conf_thres=conf)
    raise ValueError(f"Неизвестная модель: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Инференс детектора людей.")
    parser.add_argument("--model", choices=["rtdetr", "yolov8"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--warmup", type=int, default=30)
    return parser.parse_args()


def maybe_save_json(path: Optional[str], payload: dict) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    detector = build_detector(args.model, device=device, conf=args.conf)

    cap, fps, width, height, n_frames = open_video(args.input)
    writer = make_writer(args.output, fps, width, height)
    bench = Benchmark(warmup=args.warmup)

    total = n_frames
    if args.max_frames is not None:
        total = min(total, args.max_frames) if total else args.max_frames

    try:
        for idx, frame in tqdm(iter_frames(cap), total=total):
            if args.max_frames is not None and idx >= args.max_frames:
                break
            start = time.perf_counter()
            detections = detector.predict(frame)
            draw_detections(frame, detections)
            writer.write(frame)
            bench.update(time.perf_counter() - start)
    finally:
        cap.release()
        writer.release()

    metrics = bench.summary()
    metrics.update(
        {
            "model": args.model,
            "device": device,
            "conf": args.conf,
            "input": args.input,
            "output": args.output,
            "warmup": args.warmup,
        }
    )
    if args.max_frames is not None:
        metrics["max_frames"] = args.max_frames

    maybe_save_json(args.save_json, metrics)


if __name__ == "__main__":
    main()
