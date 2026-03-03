from pathlib import Path
import csv
import random
import shutil
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


DATA_DIR = Path(r"\\intram\Paris\Labo\IBHGC\PROJETS\metasport\DATA")
OUT_DIR = Path("./yolo_dataset1")

MODE = "detect"  # "detect" ou "seg"
MODEL_PATH = "yolo26x.pt" if MODE == "detect" else "yolo26x-seg.pt"

# Mets ici les noms EXACTS des dossiers sujets sous DATA_DIR
SUBJECT_NAMES = ["METASPORT10", "METASPORT11", "METASPORT12"]

SESSION_NAME = "session1"
FILES_NAME = "files"
VIDEO_FOLDER_NAME = "fichiers_video"
BRUTS_NAME = "Bruts"

BRUTS_DIRS = [
    DATA_DIR / s / SESSION_NAME / FILES_NAME / VIDEO_FOLDER_NAME / BRUTS_NAME
    for s in SUBJECT_NAMES
]

STRIDE = 10
MAX_PER_VIDEO = 0
RESIZE_TO = None

DEVICE = "0"
#IMGSZ = 960
IMGSZ = 640
CONF = 0.25

PERSON_CLASS_IN_MODEL = 0
WRITE_CLASS_ID = 0
KEEP_EMPTY_IMAGES = False
TARGET_TRIAL_SUFFIX = "_03"
EXPECTED_VIEWS = 4

VAL_RATIO = 0.15
SEED = 0

APPROX_EPS_FRAC = 0.002
MIN_AREA_PX = 80.0

PREVIEW_N = 60
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def clear_dir(p: Path) -> None:
    if not p.exists():
        return
    for f in p.glob("*"):
        if f.is_file():
            f.unlink()
        else:
            shutil.rmtree(f)


def reset_output(out_root: Path) -> None:
    for sub in [
        out_root / "images" / "all",
        out_root / "labels" / "all",
        out_root / "images" / "train",
        out_root / "images" / "val",
        out_root / "labels" / "train",
        out_root / "labels" / "val",
        out_root / "preview",
    ]:
        sub.mkdir(parents=True, exist_ok=True)
        clear_dir(sub)


def safe_imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def trial_from_video_name(vp: Path) -> str:
    # CMJump_01.xxxxxx.avi -> CMJump_01
    return vp.name.split(".", 1)[0]


# def list_videos_from_bruts_dirs(bruts_dirs: List[Path]) -> List[Tuple[Path, str]]:
#     out: List[Tuple[Path, str]] = []
#     for br in bruts_dirs:
#         if not br.exists():
#             print(f"[WARN] Missing Bruts dir: {br}")
#             continue

#         videos = sorted(br.glob("*.avi"))
#         if not videos:
#             print(f"[WARN] No .avi in {br}")
#             continue

#         by_trial = {}
#         for v in videos:
#             t = trial_from_video_name(v)
#             by_trial.setdefault(t, []).append(v)

#         for trial, vs in sorted(by_trial.items()):
#             if len(vs) != 4:
#                 print(f"[WARN] {br}: trial={trial} -> {len(vs)} video(s) (expected 4)")
#             for v in sorted(vs):
#                 out.append((v, trial))

#     seen = set()
#     dedup = []
#     for vp, trial in out:
#         if vp in seen:
#             continue
#         seen.add(vp)
#         dedup.append((vp, trial))

#     print(f"[INFO] Selected videos: {len(dedup)}")
#     return dedup

def list_videos_from_bruts_dirs(bruts_dirs: List[Path]) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []

    for br in bruts_dirs:
        if not br.exists():
            print(f"[WARN] Missing Bruts dir: {br}")
            continue

        videos = sorted(br.glob("*.avi"))
        if not videos:
            print(f"[WARN] No .avi in {br}")
            continue

        by_trial = {}
        for v in videos:
            t = trial_from_video_name(v)  # ex: CMJump_03
            if not t.endswith(TARGET_TRIAL_SUFFIX):
                continue
            by_trial.setdefault(t, []).append(v)

        for trial, vs in sorted(by_trial.items()):
            vs = sorted(vs)

            if len(vs) < EXPECTED_VIEWS:
                print(f"[WARN] {br}: trial={trial} -> {len(vs)} video(s) (expected {EXPECTED_VIEWS}) -> skipped")
                continue

            if len(vs) > EXPECTED_VIEWS:
                print(f"[WARN] {br}: trial={trial} -> {len(vs)} video(s) (expected {EXPECTED_VIEWS}) -> keeping first {EXPECTED_VIEWS}")
                vs = vs[:EXPECTED_VIEWS]

            for v in vs:
                out.append((v, trial))

    # dédoublonnage
    seen = set()
    dedup = []
    for vp, trial in out:
        if vp in seen:
            continue
        seen.add(vp)
        dedup.append((vp, trial))

    print(f"[INFO] Selected videos: {len(dedup)}")
    return dedup


def extract_frames(videos: List[Tuple[Path, str]], out_img_dir: Path, manifest_path: Path) -> None:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "video", "frame_idx", "trial"])

        total = 0
        for vp, trial in videos:
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                print(f"[WARN] Cannot open {vp}")
                continue

            try:
                rel = vp.relative_to(DATA_DIR).with_suffix("")
                stem_unique = "__".join(rel.parts)
            except ValueError:
                stem_unique = vp.with_suffix("").name

            idx = 0
            saved = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if idx % STRIDE == 0:
                    if RESIZE_TO is not None:
                        frame = cv2.resize(frame, RESIZE_TO, interpolation=cv2.INTER_AREA)

                    img_path = out_img_dir / f"{stem_unique}_f{idx:06d}.jpg"
                    safe_imwrite(img_path, frame)
                    w.writerow([str(img_path), str(vp), idx, trial])
                    saved += 1
                    total += 1

                    if MAX_PER_VIDEO and saved >= MAX_PER_VIDEO:
                        break

                idx += 1

            cap.release()
            print(f"[OK] {vp.name} ({trial}) -> {saved} frames")

    print(f"[OK] Total extracted images: {total}")


def write_yolo_detect_label(txt_path: Path, xywhn: np.ndarray, class_id: int = 0) -> None:
    lines = [f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in xywhn]
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def mask_to_polygon_yolo(mask: np.ndarray, img_w: int, img_h: int) -> Optional[np.ndarray]:
    m = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(cnt)) < MIN_AREA_PX:
        return None
    eps = APPROX_EPS_FRAC * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    if approx.shape[0] < 3:
        return None
    pts = approx.astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0] / float(img_w), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(img_h), 0.0, 1.0)
    return pts


def write_yolo_seg_label(txt_path: Path, polygons: List[np.ndarray], class_id: int = 0) -> None:
    lines = []
    for pts in polygons:
        flat = pts.reshape(-1)
        lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in flat))
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def pseudo_label(img_dir: Path, lbl_dir: Path) -> None:
    lbl_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=str(img_dir),
        imgsz=IMGSZ,
        conf=CONF,
        device=DEVICE,
        verbose=False,
        stream=True,
    )

    n_imgs = 0
    n_labeled = 0
    n_empty = 0

    for res in results:
        n_imgs += 1
        img_path = Path(res.path)
        out_txt = lbl_dir / f"{img_path.stem}.txt"

        if MODE == "detect":
            if res.boxes is None or len(res.boxes) == 0:
                n_empty += 1
                if KEEP_EMPTY_IMAGES:
                    out_txt.write_text("", encoding="utf-8")
                else:
                    img_path.unlink(missing_ok=True)
                continue

            cls = res.boxes.cls.cpu().numpy().astype(int)
            keep = (cls == PERSON_CLASS_IN_MODEL)
            if not np.any(keep):
                n_empty += 1
                if KEEP_EMPTY_IMAGES:
                    out_txt.write_text("", encoding="utf-8")
                else:
                    img_path.unlink(missing_ok=True)
                continue

            xywhn = res.boxes.xywhn.cpu().numpy()[keep]
            write_yolo_detect_label(out_txt, xywhn, class_id=WRITE_CLASS_ID)
            n_labeled += 1

        else:
            if res.masks is None or res.boxes is None or len(res.boxes) == 0:
                n_empty += 1
                if KEEP_EMPTY_IMAGES:
                    out_txt.write_text("", encoding="utf-8")
                else:
                    img_path.unlink(missing_ok=True)
                continue

            h, w = res.orig_shape
            cls = res.boxes.cls.cpu().numpy().astype(int)
            keep = (cls == PERSON_CLASS_IN_MODEL)
            if not np.any(keep):
                n_empty += 1
                if KEEP_EMPTY_IMAGES:
                    out_txt.write_text("", encoding="utf-8")
                else:
                    img_path.unlink(missing_ok=True)
                continue

            polygons: List[np.ndarray] = []
            masks = res.masks.data.cpu().numpy()
            for k in np.where(keep)[0]:
                pts = mask_to_polygon_yolo(masks[k], img_w=w, img_h=h)
                if pts is not None:
                    polygons.append(pts)

            if polygons:
                write_yolo_seg_label(out_txt, polygons, class_id=WRITE_CLASS_ID)
                n_labeled += 1
            else:
                n_empty += 1
                if KEEP_EMPTY_IMAGES:
                    out_txt.write_text("", encoding="utf-8")
                else:
                    img_path.unlink(missing_ok=True)

        if n_imgs % 200 == 0:
            print(f"[...] {n_imgs} | labeled={n_labeled} | empty={n_empty}")

    print(f"[OK] Pseudo-label: processed={n_imgs}, labeled={n_labeled}, empty={n_empty}")


def split_train_val(out_root: Path) -> None:
    img_all = out_root / "images" / "all"
    lbl_all = out_root / "labels" / "all"

    img_train = out_root / "images" / "train"
    img_val = out_root / "images" / "val"
    lbl_train = out_root / "labels" / "train"
    lbl_val = out_root / "labels" / "val"

    for p in [img_train, img_val, lbl_train, lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in img_all.iterdir() if p.suffix.lower() in IMG_EXTS])
    pairs = [(img, lbl_all / f"{img.stem}.txt") for img in images if (lbl_all / f"{img.stem}.txt").exists()]

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    n_val = int(len(pairs) * VAL_RATIO)

    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    def move_pairs(pairs_list, dst_i: Path, dst_l: Path):
        for img, lbl in pairs_list:
            shutil.move(str(img), str(dst_i / img.name))
            shutil.move(str(lbl), str(dst_l / lbl.name))

    move_pairs(train_pairs, img_train, lbl_train)
    move_pairs(val_pairs, img_val, lbl_val)
    print(f"[OK] Split: train={len(train_pairs)}, val={len(val_pairs)}")


def write_dataset_yaml(out_root: Path) -> None:
    yaml_path = out_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: person\n",
        encoding="utf-8",
    )
    print(f"[OK] dataset.yaml: {yaml_path}")


def preview_samples(out_root: Path) -> None:
    if PREVIEW_N <= 0:
        return

    img_train = out_root / "images" / "train"
    lbl_train = out_root / "labels" / "train"
    prev_dir = out_root / "preview"
    prev_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in img_train.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        return

    rng = random.Random(SEED)
    sample = rng.sample(images, k=min(PREVIEW_N, len(images)))

    for img_path in sample:
        lbl_path = lbl_train / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()

            if MODE == "detect":
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            else:
                if len(parts) < 7:
                    continue
                coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
                pts = np.stack([coords[:, 0] * w, coords[:, 1] * h], axis=1).astype(np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imwrite(str(prev_dir / f"{img_path.stem}_preview.jpg"), img)

    print(f"[OK] Preview: {prev_dir}")


def main():
    print(f"[INFO] MODE={MODE} MODEL={MODEL_PATH}")
    if not Path(MODEL_PATH).exists():
        print(f"[WARN] Weights not found: {MODEL_PATH} ")

    reset_output(OUT_DIR)

    videos = list_videos_from_bruts_dirs(BRUTS_DIRS)
    if not videos:
        raise SystemExit("No videos found. Check SUBJECT_NAMES / folder names and Bruts path.")

    img_all = OUT_DIR / "images" / "all"
    lbl_all = OUT_DIR / "labels" / "all"
    manifest = OUT_DIR / "manifest.csv"

    extract_frames(videos, img_all, manifest)
    pseudo_label(img_all, lbl_all)
    split_train_val(OUT_DIR)
    write_dataset_yaml(OUT_DIR)
    preview_samples(OUT_DIR)

    print(f"[OK] Dataset: {OUT_DIR / 'dataset.yaml'}")
    if MODE == "detect":
        print(f"yolo detect train model={MODEL_PATH} data={OUT_DIR / 'dataset.yaml'} epochs=80 imgsz={IMGSZ} device={DEVICE}")
    else:
        print(f"yolo segment train model={MODEL_PATH} data={OUT_DIR / 'dataset.yaml'} epochs=80 imgsz={IMGSZ} device={DEVICE}")


if __name__ == "__main__":
    main()