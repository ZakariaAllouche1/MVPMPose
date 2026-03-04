import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from boxmot import DeepOcSort

# Patch BoxMOT: évite "float(h)" quand h est un array (1,)
from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR
from copy import deepcopy
from collections import deque


def _to_scalar(x):
    x = np.asarray(x).squeeze()
    return float(x)


def _patched_unfreeze(self):
    if self.attr_saved is None:
        return

    new_history = deepcopy(list(self.history_obs))

    self.__dict__ = self.attr_saved
    self.history_obs = deque(list(self.history_obs)[:-1], maxlen=self.max_obs)

    occur = [int(d is None) for d in new_history]
    indices = np.where(np.array(occur) == 0)[0]
    if len(indices) < 2:
        return

    index1, index2 = int(indices[-2]), int(indices[-1])
    if index2 <= index1:
        return

    box1, box2 = new_history[index1], new_history[index2]

    x1, y1, s1, r1 = box1
    w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)

    x2, y2, s2, r2 = box2
    w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)

    time_gap = index2 - index1
    dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
    dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap

    for i in range(time_gap):
        x = x1 + (i + 1) * dx
        y = y1 + (i + 1) * dy
        w = w1 + (i + 1) * dw
        h = h1 + (i + 1) * dh

        w_s = _to_scalar(w)
        h_s = _to_scalar(h)

        s = w_s * h_s
        r = w_s / h_s

        new_box = np.array([_to_scalar(x), _to_scalar(y), s, r], dtype=np.float32).reshape((4, 1))

        self.update(new_box)

        if i != (time_gap - 1):
            self.predict()
            self.history_obs.pop()
            self.history_obs.pop()


KalmanFilterXYSR.unfreeze = _patched_unfreeze


def test_boxmot_gpu():
    video_path = r"\\intram\Paris\Labo\IBHGC\PROJETS\metasport\DATA\METASPORT11\Session1\Files\fichiers_video\Bruts\CMJump_03.51433950.20251114140826.avi"
    model_path = "yolo26x.pt"

    yolo_model = YOLO(model_path)

    tracker = DeepOcSort(
        reid_weights=Path("osnet_x0_25_msmt17.pt"),
        device=torch.device("cuda:0"),
        half=True,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur: impossible de lire la video")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps0 = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps0) if fps0 and fps0 == fps0 and fps0 > 0 else 30

    out = cv2.VideoWriter("sujet11_jump3.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        res = yolo_model.predict(frame, classes=[0], conf=0.3, verbose=False, device="cuda:0")

        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            boxes = res[0].boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            confs = res[0].boxes.conf.detach().cpu().numpy().astype(np.float32).reshape(-1)
            clss = res[0].boxes.cls.detach().cpu().numpy().astype(np.float32).reshape(-1)

            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)

            dets = np.column_stack((boxes, confs, clss)).astype(np.float32)
            dets = np.ascontiguousarray(dets)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        tracks = tracker.update(dets, frame)

        if tracks is not None and len(tracks) > 0:
            for t in tracks:
                x1, y1, x2, y2, track_id = map(int, t[:5])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"{frame_count} frames")

    cap.release()
    out.release()
    print("Termine")


if __name__ == "__main__":
    test_boxmot_gpu()