from __future__ import annotations

from typing import List

from ultralytics import RTDETR

from app.detectors.base import BaseDetector
from app.types import Detection

PERSON_CLASS_ID = 0


class RTDETRDetector(BaseDetector):
    def __init__(self, device: str, conf_thres: float, weights: str = "rtdetr-l.pt") -> None:
        super().__init__(device, conf_thres)
        self.model = RTDETR(weights)

    def predict(self, frame_bgr) -> List[Detection]:
        results = self.model.predict(
            source=frame_bgr,
            verbose=False,
            device=self.device,
            conf=self.conf_thres,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        labels = boxes.cls.cpu().numpy().astype(int)
        detections: List[Detection] = []
        for box, score, label in zip(xyxy, scores, labels):
            if label != PERSON_CLASS_ID:
                continue
            if score < self.conf_thres:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append(Detection(x1, y1, x2, y2, float(score)))
        return detections
