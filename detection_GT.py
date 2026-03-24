#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import csv
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = "./runs/detect/finetune_undistort/weights/best.pt"
SUBJECT_VIDEOS_DIR = "./metasport/DATA/METASPORT12_undistort/Session1/Files/fichiers_video/Bruts/METASPORT12_undistorted"
OUTPUT_DATASET_DIR = "./detection_eval_subject12_cvat"

FRAME_STEP = 10
CONF_THRESH = 0.25
PERSON_CLASS_ID = 0

SUBSET_NAME = "val"
CLASS_NAMES = {0: "person"}

# Options
JPEG_QUALITY = 90
SAVE_PREVIEW = False
SAVE_CSV = False
RESET_OUTPUT_DIR = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def save_label_yolo(txt_path: Path, boxes_xyxy, img_w, img_h, class_id=0):
    with open(txt_path, "w", encoding="utf-8") as f:
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
    return Path(xcp_path).stem


def get_activity_root(output_root: Path, activity_name: str) -> Path:
    return output_root / activity_name


def init_activity_structure(activity_root: Path):
    ensure_dir(activity_root / "images" / SUBSET_NAME)
    ensure_dir(activity_root / "labels" / SUBSET_NAME)

    if SAVE_PREVIEW:
        ensure_dir(activity_root / "preview")

    if SAVE_CSV:
        ensure_dir(activity_root / "metadata")


def write_data_yaml(activity_root: Path, subset_name: str, class_names: dict):
    yaml_path = activity_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: ./\n")
        f.write(f"{subset_name}: {subset_name}.txt\n")
        f.write("names:\n")
        for class_id in sorted(class_names.keys()):
            f.write(f"  {class_id}: {class_names[class_id]}\n")
    return yaml_path


def write_subset_txt(activity_root: Path, subset_name: str):
    images_dir = activity_root / "images" / subset_name
    subset_txt = activity_root / f"{subset_name}.txt"

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    with open(subset_txt, "w", encoding="utf-8") as f:
        for img_path in image_paths:
            f.write(f"images/{subset_name}/{img_path.name}\n")

    return subset_txt, len(image_paths)


def process_video(
    model,
    video_path: str,
    activity_name: str,
    activity_root: Path,
    frame_step: int = 10,
    conf_thresh: float = 0.25,
):
    video_stem = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir {video_path}")
        return 0

    images_dir = activity_root / "images" / SUBSET_NAME
    labels_dir = activity_root / "labels" / SUBSET_NAME

    preview_dir = activity_root / "preview" / video_stem
    if SAVE_PREVIEW:
        ensure_dir(preview_dir)

    csv_path = activity_root / "metadata" / f"{activity_name}_{video_stem}_pseudo_labels.csv"
    csv_file = None
    writer = None

    if SAVE_CSV:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow([
            "activity_name", "video_name", "image_name", "frame_idx",
            "class_id", "x1", "y1", "x2", "y2", "score"
        ])

    frame_idx = 0
    saved = 0

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

        image_path = images_dir / image_name
        label_path = labels_dir / label_name

        ok = cv2.imwrite(
            str(image_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            print(f"[WARN] Impossible d'écrire {image_path}")
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        save_label_yolo(label_path, boxes_xyxy, w, h, class_id=PERSON_CLASS_ID)

        if SAVE_PREVIEW:
            preview = draw_preview(frame, boxes_xyxy)
            cv2.imwrite(str(preview_dir / image_name), preview)

        if SAVE_CSV and writer is not None:
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

    if csv_file is not None:
        csv_file.close()

    print(f"[OK] {activity_name} | {video_stem}: {saved} images extraites")
    return saved


def build_datasets():
    output_root = Path(OUTPUT_DATASET_DIR)

    if RESET_OUTPUT_DIR:
        reset_dir(output_root)
    else:
        ensure_dir(output_root)

    model = YOLO(MODEL_PATH)

    xcp_files = sorted(glob.glob(os.path.join(SUBJECT_VIDEOS_DIR, "*.xcp")))
    if not xcp_files:
        print("[ERREUR] Aucun fichier .xcp trouvé")
        return

    grand_total_images = 0

    for xcp_path in xcp_files:
        activity_name = extract_activity_name_from_xcp(xcp_path)
        activity_root = get_activity_root(output_root, activity_name)
        init_activity_structure(activity_root)

        video_files = sorted(glob.glob(os.path.join(SUBJECT_VIDEOS_DIR, f"{activity_name}.*.mp4")))
        if not video_files:
            video_files = sorted(glob.glob(os.path.join(SUBJECT_VIDEOS_DIR, f"{activity_name}.*.avi")))

        if len(video_files) < 1:
            print(f"[WARN] Aucune vidéo trouvée pour {activity_name}")
            continue

        print("=" * 80)
        print(f"[INFO] Activité: {activity_name}")
        print(f"[INFO] Nombre de vidéos: {len(video_files)}")

        activity_total_images = 0

        for video_path in video_files:
            activity_total_images += process_video(
                model=model,
                video_path=video_path,
                activity_name=activity_name,
                activity_root=activity_root,
                frame_step=FRAME_STEP,
                conf_thresh=CONF_THRESH
            )

        yaml_path = write_data_yaml(activity_root, SUBSET_NAME, CLASS_NAMES)
        subset_txt_path, n_manifest = write_subset_txt(activity_root, SUBSET_NAME)

        print(f"[OK] Dataset CVAT créé pour {activity_name}")
        print(f"[OK] Dossier: {activity_root}")
        print(f"[OK] Images: {activity_total_images}")
        print(f"[OK] YAML: {yaml_path}")
        print(f"[OK] {SUBSET_NAME}.txt: {subset_txt_path} ({n_manifest} entrées)")

        grand_total_images += activity_total_images

    print("=" * 80)
    print(f"[OK] Tous les datasets ont été générés dans: {output_root}")
    print(f"[OK] Nombre total d'images: {grand_total_images}")


if __name__ == "__main__":
    build_datasets()