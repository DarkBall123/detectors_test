from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.types import Detection


class BaseDetector(ABC):
    def __init__(self, device: str, conf_thres: float) -> None:
        self.device = device
        self.conf_thres = conf_thres

    @abstractmethod
    def predict(self, frame_bgr) -> List[Detection]:
        raise NotImplementedError
