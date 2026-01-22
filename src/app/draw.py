from __future__ import annotations

from typing import Iterable

import cv2

from app.types import Detection


def draw_detections(frame, detections: Iterable[Detection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 220, 20), 2)
        label = f"person {det.score:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 220, 20),
            1,
            cv2.LINE_AA,
        )
