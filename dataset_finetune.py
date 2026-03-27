from pathlib import Path
import csv
import random
import shutil
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Dossier racine des données
DATA_DIR = Path("./metasport/DATA")

# Dossier de sortie du dataset YOLO
OUT_DIR = Path("./yolo_dataset_clean")

# Modèle de base utilisé pour les pseudo-labels
MODEL_PATH = "yolo26x.pt"

# Sujets utilisés pour le train
TRAIN_SUBJECTS = {
    "METASPORT01",
    "METASPORT02",
    "METASPORT03",
    "METASPORT04",
    "METASPORT05",
    "METASPORT07",
    "METASPORT08",
    "METASPORT11",
}

# Sujets utilisés pour la validation
VAL_SUBJECTS = {
    "METASPORT09",
    "METASPORT10",
}

# Sujets complètement exclus pour test externe
HOLDOUT_SUBJECTS = {
    "METASPORT06",
    "METASPORT12",
}

# Activités à garder
ALLOWED_TRIAL_SUFFIXES: Optional[Set[str]] = {
    "_03",
    "_01",
    "_02",
    # "_04",
}

SESSION_NAME = "session1"
FILES_NAME = "files"
VIDEO_FOLDER_NAME = "fichiers_video"
BRUTS_NAME = "Bruts"

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

# Extraction des frames
STRIDE = 10
MAX_PER_VIDEO = 0
RESIZE_TO = None

# Paramètres du pseudo-label
DEVICE = "0"
PSEUDO_IMGSZ = 1280
CONF = 0.20
PERSON_CLASS_IN_MODEL = 0
WRITE_CLASS_ID = 0
KEEP_EMPTY_IMAGES = False

# Paramètres généraux
EXPECTED_VIEWS = 4
SEED = 0

# Nombre d'images à prévisualiser
PREVIEW_N_TRAIN = 120
PREVIEW_N_VAL = 60

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
        out_root / "preview" / "train",
        out_root / "preview" / "val",
    ]:
        sub.mkdir(parents=True, exist_ok=True)
        clear_dir(sub)

    for f in [
        out_root / "manifest.csv",
        out_root / "train.txt",
        out_root / "val.txt",
        out_root / "dataset.yaml",
    ]:
        if f.exists():
            f.unlink()


def safe_imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def trial_from_video_name(vp: Path) -> str:
    # Exemple : CMJump_03.xxxxxx.avi -> CMJump_03
    return vp.name.split(".", 1)[0]


def subject_bruts_dir(subject_name: str) -> Path:
    return (
        DATA_DIR
        / subject_name
        / SESSION_NAME
        / FILES_NAME
        / VIDEO_FOLDER_NAME
        / BRUTS_NAME
    )


def list_subject_video_dirs(subjects: Set[str]) -> List[Tuple[str, Path]]:
    return [(s, subject_bruts_dir(s)) for s in sorted(subjects)]


def is_allowed_trial(trial_name: str) -> bool:
    if ALLOWED_TRIAL_SUFFIXES is None:
        return True
    return any(trial_name.endswith(sfx) for sfx in ALLOWED_TRIAL_SUFFIXES)


def list_videos_from_subjects(subjects: Set[str]) -> List[Tuple[Path, str, str]]:
    # Retourne : (chemin vidéo, nom activité/trial, sujet)
    out: List[Tuple[Path, str, str]] = []

    for subject, br in list_subject_video_dirs(subjects):
        if not br.exists():
            print(f"[WARN] Missing Bruts dir: {br}")
            continue

        videos = sorted([p for p in br.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
        if not videos:
            print(f"[WARN] No videos in {br}")
            continue

        by_trial: Dict[str, List[Path]] = {}
        for v in videos:
            t = trial_from_video_name(v)
            if not is_allowed_trial(t):
                continue
            by_trial.setdefault(t, []).append(v)

        for trial, vs in sorted(by_trial.items()):
            vs = sorted(vs)

            if len(vs) < EXPECTED_VIEWS:
                print(f"[WARN] {subject} | trial={trial} -> {len(vs)} video(s), skipped")
                continue

            if len(vs) > EXPECTED_VIEWS:
                print(f"[WARN] {subject} | trial={trial} -> {len(vs)} video(s), keeping first {EXPECTED_VIEWS}")
                vs = vs[:EXPECTED_VIEWS]

            for v in vs:
                out.append((v, trial, subject))

    # Supprime les doublons éventuels
    seen = set()
    dedup = []
    for vp, trial, subject in out:
        if vp in seen:
            continue
        seen.add(vp)
        dedup.append((vp, trial, subject))

    print(f"[INFO] Selected videos: {len(dedup)}")
    return dedup


def preprocess_frame(
    frame: np.ndarray,
    subject: str,
    trial: str,
    video_path: Path,
    frame_idx: int,
) -> np.ndarray:
    # Si tes vidéos sont déjà undistorted + floutées, ne change rien ici
    return frame


def extract_frames(videos: List[Tuple[Path, str, str]], out_img_dir: Path, manifest_path: Path) -> None:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "video", "frame_idx", "trial", "subject"])

        total = 0
        for vp, trial, subject in videos:
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
                    frame = preprocess_frame(frame, subject, trial, vp, idx)

                    if RESIZE_TO is not None:
                        frame = cv2.resize(frame, RESIZE_TO, interpolation=cv2.INTER_AREA)

                    img_path = out_img_dir / f"{stem_unique}_f{idx:06d}.jpg"
                    safe_imwrite(img_path, frame)

                    w.writerow([
                        str(img_path.resolve()),
                        str(vp),
                        idx,
                        trial,
                        subject,
                    ])

                    saved += 1
                    total += 1

                    if MAX_PER_VIDEO and saved >= MAX_PER_VIDEO:
                        break

                idx += 1

            cap.release()
            print(f"[OK] {subject} | {vp.name} ({trial}) -> {saved} frames")

    print(f"[OK] Total extracted images: {total}")


def write_yolo_detect_label(txt_path: Path, xywhn: np.ndarray, class_id: int = 0) -> None:
    # Format YOLO detect : class cx cy w h
    lines = [f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in xywhn]
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def pseudo_label(img_dir: Path, lbl_dir: Path) -> None:
    lbl_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=str(img_dir),
        imgsz=PSEUDO_IMGSZ,
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

        if n_imgs % 200 == 0:
            print(f"[...] {n_imgs} | labeled={n_labeled} | empty={n_empty}")

    print(f"[OK] Pseudo-label: processed={n_imgs}, labeled={n_labeled}, empty={n_empty}")


def write_train_val_txt(out_root: Path) -> None:
    # Crée train.txt et val.txt à partir des sujets choisis
    manifest_path = out_root / "manifest.csv"
    lbl_all = out_root / "labels" / "all"

    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    filtered_rows = []
    for r in rows:
        img_path = Path(r["image"])
        lbl_path = lbl_all / f"{img_path.stem}.txt"
        if img_path.exists() and lbl_path.exists():
            filtered_rows.append(r)

    train_rows = [r for r in filtered_rows if r["subject"] in TRAIN_SUBJECTS]
    val_rows = [r for r in filtered_rows if r["subject"] in VAL_SUBJECTS]

    rng = random.Random(SEED)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    (out_root / "train.txt").write_text(
        "\n".join(r["image"] for r in train_rows),
        encoding="utf-8",
    )
    (out_root / "val.txt").write_text(
        "\n".join(r["image"] for r in val_rows),
        encoding="utf-8",
    )

    print(f"[OK] train.txt images: {len(train_rows)}")
    print(f"[OK] val.txt images:   {len(val_rows)}")


def write_dataset_yaml(out_root: Path) -> None:
    yaml_path = out_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_root.resolve().as_posix()}\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "names:\n"
        "  0: person\n",
        encoding="utf-8",
    )
    print(f"[OK] dataset.yaml: {yaml_path}")


def draw_boxes_on_image(img: np.ndarray, lbl_path: Path) -> np.ndarray:
    h, w = img.shape[:2]

    text = lbl_path.read_text(encoding="utf-8").strip()
    if not text:
        return img

    for line in text.splitlines():
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    return img


def preview_from_txt(out_root: Path, split_name: str, n_preview: int) -> None:
    # Génère quelques images de preview avec bbox
    txt_file = out_root / f"{split_name}.txt"
    lbl_all = out_root / "labels" / "all"
    prev_dir = out_root / "preview" / split_name
    prev_dir.mkdir(parents=True, exist_ok=True)

    if not txt_file.exists():
        print(f"[WARN] Missing {txt_file}")
        return

    images = [Path(x) for x in txt_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not images:
        print(f"[WARN] No images in {txt_file}")
        return

    rng = random.Random(SEED)
    sample = rng.sample(images, k=min(n_preview, len(images)))

    count = 0
    for img_path in sample:
        lbl_path = lbl_all / f"{img_path.stem}.txt"
        if not img_path.exists() or not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img = draw_boxes_on_image(img, lbl_path)
        out_path = prev_dir / f"{img_path.stem}_preview.jpg"
        cv2.imwrite(str(out_path), img)
        count += 1

    print(f"[OK] Preview {split_name}: {count} images -> {prev_dir}")


def validate_config() -> None:
    # Vérifie qu'il n'y a pas de chevauchement entre les groupes de sujets
    if TRAIN_SUBJECTS & VAL_SUBJECTS:
        raise ValueError(f"Overlap TRAIN/VAL: {TRAIN_SUBJECTS & VAL_SUBJECTS}")
    if TRAIN_SUBJECTS & HOLDOUT_SUBJECTS:
        raise ValueError(f"Overlap TRAIN/HOLDOUT: {TRAIN_SUBJECTS & HOLDOUT_SUBJECTS}")
    if VAL_SUBJECTS & HOLDOUT_SUBJECTS:
        raise ValueError(f"Overlap VAL/HOLDOUT: {VAL_SUBJECTS & HOLDOUT_SUBJECTS}")
    if not TRAIN_SUBJECTS:
        raise ValueError("TRAIN_SUBJECTS is empty")
    if not VAL_SUBJECTS:
        raise ValueError("VAL_SUBJECTS is empty")


def main():
    validate_config()

    used_subjects = TRAIN_SUBJECTS | VAL_SUBJECTS

    print(f"[INFO] MODEL={MODEL_PATH}")
    print(f"[INFO] Train subjects: {sorted(TRAIN_SUBJECTS)}")
    print(f"[INFO] Val subjects:   {sorted(VAL_SUBJECTS)}")
    print(f"[INFO] Holdout:        {sorted(HOLDOUT_SUBJECTS)}")
    print(f"[INFO] Allowed trials: {sorted(ALLOWED_TRIAL_SUFFIXES) if ALLOWED_TRIAL_SUFFIXES is not None else 'ALL'}")

    if not Path(MODEL_PATH).exists():
        print(f"[WARN] Weights not found locally: {MODEL_PATH}")

    reset_output(OUT_DIR)

    videos = list_videos_from_subjects(used_subjects)
    if not videos:
        raise SystemExit("No videos found. Check subject names, folders, and activity filters.")

    img_all = OUT_DIR / "images" / "all"
    lbl_all = OUT_DIR / "labels" / "all"
    manifest = OUT_DIR / "manifest.csv"

    extract_frames(videos, img_all, manifest)
    pseudo_label(img_all, lbl_all)
    write_train_val_txt(OUT_DIR)
    write_dataset_yaml(OUT_DIR)

    if PREVIEW_N_TRAIN > 0:
        preview_from_txt(OUT_DIR, "train", PREVIEW_N_TRAIN)
    if PREVIEW_N_VAL > 0:
        preview_from_txt(OUT_DIR, "val", PREVIEW_N_VAL)

    print(f"[OK] Dataset YAML: {OUT_DIR / 'dataset.yaml'}")
    print(
        "Suggested train command:\n"
        f"yolo detect train model={MODEL_PATH} data={OUT_DIR / 'dataset.yaml'} "
        "epochs=200 patience=0 imgsz=1280 optimizer=auto batch=0.70 device=0 "
        "workers=8 cache=ram plots=True save_period=10"
    )


if __name__ == "__main__":
    main()