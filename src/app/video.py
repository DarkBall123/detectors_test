from __future__ import annotations

from typing import Iterator, Optional, Tuple

import cv2


def open_video(path: str) -> Tuple[cv2.VideoCapture, float, int, int, Optional[int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    n_frames = int(frame_count) if frame_count and frame_count > 0 else None
    return cap, fps, width, height, n_frames


def make_writer(
    path: str, fps: float, width: int, height: int, codec: str = "mp4v"
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось открыть видео для записи: {path}")
    return writer


def iter_frames(cap: cv2.VideoCapture) -> Iterator[tuple[int, "cv2.Mat"]]:
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1
