"""
Patch torchvision.io.read_video for torchvision >= 0.26 which removed it.
Import this module before any code that uses `from torchvision.io import read_video`.
"""

import cv2
import torch
import numpy as np
import torchvision.io


def read_video(filename, start_pts=0, end_pts=None, pts_unit="sec"):
    """Read video using OpenCV, returns (video, audio, info) matching old API."""
    cap = cv2.VideoCapture(str(filename))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        video = torch.zeros(0, 0, 0, 3, dtype=torch.uint8)
    else:
        video = torch.from_numpy(np.stack(frames))  # (T, H, W, C)

    info = {"video_fps": fps}
    return video, None, info


# Monkey-patch into torchvision.io
torchvision.io.read_video = read_video
