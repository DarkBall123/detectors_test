from __future__ import annotations

import argparse
import os
import shutil

from ultralytics import RTDETR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Экспорт RT-DETR в ONNX.")
    parser.add_argument("--weights", default="rtdetr-l.pt")
    parser.add_argument("--output", default="artifacts/rtdetr.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = RTDETR(args.weights)
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
    )
    export_path = str(exported)
    output = args.output
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if export_path != output:
        shutil.copy(export_path, output)


if __name__ == "__main__":
    main()
