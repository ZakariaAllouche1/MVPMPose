#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import csv
import cv2
from ultralytics import YOLO


# =========================
# Réglages à modifier
# =========================
MODEL_PATH = "./runs/detect/finetune_undistort/weights/best.pt"
SUBJECT_VIDEOS_DIR = "./metasport/DATA/METASPORT12_undistort/Session1/Files/fichiers_video/Bruts/METASPORT12_undistorted"
OUTPUT_DATASET_DIR = "./detection_eval_subject12"
FRAME_STEP = 10
CONF_THRESH = 0.25
PERSON_CLASS_ID = 0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def save_label_yolo(txt_path, boxes_xyxy, img_w, img_h, class_id=0):
    with open(txt_path, "w") as f:
        for x1, y1, x2, y2, _score in boxes_xyxy:
            cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def draw_preview(img, boxes_xyxy):
    vis = img.copy()
    for x1, y1, x2, y2, score in boxes_xyxy:
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{score:.2f}",
            (x1i, max(20, y1i - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
    return vis


def extract_activity_name_from_xcp(xcp_path: str) -> str:
    return os.path.basename(xcp_path).replace(".xcp", "")


def process_video(model, video_path, activity_name, output_root, frame_step=10, conf_thresh=0.25):
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir {video_path}")
        return 0

    images_dir = os.path.join(output_root, "images", "val")
    labels_dir = os.path.join(output_root, "labels", "val")
    preview_dir = os.path.join(output_root, "preview", activity_name)

    ensure_dir(images_dir)
    ensure_dir(labels_dir)
    ensure_dir(preview_dir)

    frame_idx = 0
    saved = 0

    csv_path = os.path.join(output_root, f"{activity_name}_{video_stem}_pseudo_labels.csv")
    csv_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not csv_exists:
            writer.writerow([
                "activity_name", "video_name", "image_name", "frame_idx",
                "class_id", "x1", "y1", "x2", "y2", "score"
            ])

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            results = model.predict(
                source=frame,
                conf=conf_thresh,
                classes=[PERSON_CLASS_ID],
                verbose=False
            )

            boxes_xyxy = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                    score = float(box.conf[0])
                    boxes_xyxy.append((x1, y1, x2, y2, score))

            image_name = f"{activity_name}__{video_stem}__f{frame_idx:06d}.jpg"
            label_name = f"{activity_name}__{video_stem}__f{frame_idx:06d}.txt"

            image_path = os.path.join(images_dir, image_name)
            label_path = os.path.join(labels_dir, label_name)
            preview_path = os.path.join(preview_dir, image_name)

            cv2.imwrite(image_path, frame)
            h, w = frame.shape[:2]
            save_label_yolo(label_path, boxes_xyxy, w, h, class_id=PERSON_CLASS_ID)

            preview = draw_preview(frame, boxes_xyxy)
            cv2.imwrite(preview_path, preview)

            for x1, y1, x2, y2, score in boxes_xyxy:
                writer.writerow([
                    activity_name, video_stem, image_name, frame_idx,
                    PERSON_CLASS_ID,
                    round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                    round(score, 4)
                ])

            saved += 1
            frame_idx += 1

    cap.release()
    print(f"[OK] {video_stem}: {saved} images extraites")
    return saved


def build_dataset():
    ensure_dir(OUTPUT_DATASET_DIR)
    ensure_dir(os.path.join(OUTPUT_DATASET_DIR, "images", "val"))
    ensure_dir(os.path.join(OUTPUT_DATASET_DIR, "labels", "val"))
    ensure_dir(os.path.join(OUTPUT_DATASET_DIR, "preview"))

    model = YOLO(MODEL_PATH)

    xcp_files = sorted(glob.glob(os.path.join(SUBJECT_VIDEOS_DIR, "*.xcp")))
    if not xcp_files:
        print("[ERREUR] Aucun fichier .xcp trouvé")
        return

    total_images = 0

    for xcp_path in xcp_files:
        activity_name = extract_activity_name_from_xcp(xcp_path)
        video_files = sorted(glob.glob(os.path.join(SUBJECT_VIDEOS_DIR, f"{activity_name}.*.mp4")))
        if not video_files:
            video_files = sorted(glob.glob(os.path.join(SUBJECT_VIDEOS_DIR, f"{activity_name}.*.avi")))

        if len(video_files) < 1:
            print(f"[WARN] Aucune vidéo trouvée pour {activity_name}")
            continue

        print("=" * 80)
        print(f"[INFO] Activité: {activity_name}")
        print(f"[INFO] Nombre de vidéos: {len(video_files)}")

        for video_path in video_files:
            total_images += process_video(
                model=model,
                video_path=video_path,
                activity_name=activity_name,
                output_root=OUTPUT_DATASET_DIR,
                frame_step=FRAME_STEP,
                conf_thresh=CONF_THRESH
            )

    yaml_path = os.path.join(OUTPUT_DATASET_DIR, "data_detection_eval.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(OUTPUT_DATASET_DIR)}\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write("  0: person\n")

    print("=" * 80)
    print(f"[OK] Dataset généré dans: {OUTPUT_DATASET_DIR}")
    print(f"[OK] Nombre total d'images: {total_images}")
    print(f"[OK] YAML: {yaml_path}")


if __name__ == "__main__":
    build_dataset()