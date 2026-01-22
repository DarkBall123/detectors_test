from __future__ import annotations

import argparse
import json
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from app.benchmark import Benchmark
from app.draw import draw_detections
from app.types import Detection
from app.video import iter_frames, make_writer, open_video

PERSON_CLASS_ID = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Инференс RT-DETR через ONNX Runtime.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--person-id", type=int, default=PERSON_CLASS_ID)
    return parser.parse_args()


def letterbox(
    image: np.ndarray, new_shape: int, color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    height, width = image.shape[:2]
    ratio = min(new_shape / height, new_shape / width)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = new_shape - new_w
    dh = new_shape - new_h
    dw /= 2
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, ratio, (dw, dh)


def scale_boxes(
    boxes: np.ndarray, ratio: float, pad: Tuple[float, float], shape: Tuple[int, int]
) -> np.ndarray:
    boxes = boxes.copy()
    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    h, w = shape
    boxes[:, 0] = boxes[:, 0].clip(0, w - 1)
    boxes[:, 2] = boxes[:, 2].clip(0, w - 1)
    boxes[:, 1] = boxes[:, 1].clip(0, h - 1)
    boxes[:, 3] = boxes[:, 3].clip(0, h - 1)
    return boxes


def parse_outputs(raw_outputs: list[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(raw_outputs) == 4 and raw_outputs[0].size == 1:
        raw_outputs = raw_outputs[1:]
    if len(raw_outputs) >= 3:
        boxes, scores, labels = raw_outputs[:3]
        if boxes.ndim == 3:
            boxes = boxes[0]
        if scores.ndim == 2:
            scores = scores[0]
        if labels.ndim == 2:
            labels = labels[0]
        return boxes, scores, labels
    if len(raw_outputs) == 1:
        detections = raw_outputs[0]
        if detections.ndim == 3:
            detections = detections[0]
        if detections.shape[-1] >= 6:
            boxes = detections[:, :4]
            scores = detections[:, 4]
            labels = detections[:, 5]
            return boxes, scores, labels
    raise RuntimeError("Неподдерживаемый формат выходов ONNX.")


def decode_rtdetr(
    raw_output: np.ndarray, person_id: int, imgsz: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if raw_output.ndim == 3:
        raw_output = raw_output[0]
    boxes = raw_output[:, :4]
    class_scores = raw_output[:, 4:]
    scores = class_scores[:, person_id]
    labels = np.full(scores.shape, person_id, dtype=np.int64)

    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = (cx - w / 2) * imgsz
    y1 = (cy - h / 2) * imgsz
    x2 = (cx + w / 2) * imgsz
    y2 = (cy + h / 2) * imgsz
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    return xyxy, scores, labels


def maybe_save_json(path: Optional[str], payload: dict) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

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
            padded, ratio, pad = letterbox(frame, args.imgsz)
            rgb = padded[:, :, ::-1]
            blob = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32) / 255.0
            outputs = session.run(None, {input_name: blob})
            if len(outputs) == 1 and outputs[0].shape[-1] == 84:
                boxes, scores, labels = decode_rtdetr(outputs[0], args.person_id, args.imgsz)
            else:
                boxes, scores, labels = parse_outputs(outputs)
            boxes = scale_boxes(boxes, ratio, pad, (height, width))

            detections = []
            for box, score, label in zip(boxes, scores, labels):
                if int(label) != args.person_id:
                    continue
                if float(score) < args.conf:
                    continue
                x1, y1, x2, y2 = box.tolist()
                detections.append(Detection(x1, y1, x2, y2, float(score)))

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
            "device": "cpu",
            "conf": args.conf,
            "input": args.input,
            "output": args.output,
            "warmup": args.warmup,
            "imgsz": args.imgsz,
        }
    )
    if args.max_frames is not None:
        metrics["max_frames"] = args.max_frames

    maybe_save_json(args.save_json, metrics)


if __name__ == "__main__":
    main()
