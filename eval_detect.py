#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment



GT_ROOT = Path("./GT_detect_metasport12")
VIDEOS_ROOT = Path(r"\\intram\Paris\Labo\IBHGC\PROJETS\metasport\DATA\METASPORT12_undistort\Session1\Files\fichiers_video\Bruts\METASPORT12_undistorted")
PRED_ROOT = Path("./results_1/videos_eval/ResultsMETASPORT12Rtmlib_V3")
OUTPUT_DIR = Path("./eval_detection_subject12")

SUBSET = "val"
CLASS_ID = 0  # person

FRAME_OFFSET = 1

SCORE_THRESHOLD = None

ACTIVITIES = None

OPERATING_IOU = 0.50

# Pour le mAP
MAP_IOUS = np.arange(0.50, 0.96, 0.05)



# UTILS
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    x1 = cx * img_w - bw / 2.0
    y1 = cy * img_h - bh / 2.0
    x2 = x1 + bw
    y2 = y1 + bh
    return [float(x1), float(y1), float(x2), float(y2)]


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter + 1e-12
    return inter / union


def box_iou_matrix(gt_boxes, pred_boxes):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)

    gt = np.asarray(gt_boxes, dtype=np.float32)
    pr = np.asarray(pred_boxes, dtype=np.float32)

    gt_x1 = gt[:, 0:1]
    gt_y1 = gt[:, 1:2]
    gt_x2 = gt[:, 2:3]
    gt_y2 = gt[:, 3:4]

    pr_x1 = pr[:, 0]
    pr_y1 = pr[:, 1]
    pr_x2 = pr[:, 2]
    pr_y2 = pr[:, 3]

    inter_x1 = np.maximum(gt_x1, pr_x1)
    inter_y1 = np.maximum(gt_y1, pr_y1)
    inter_x2 = np.minimum(gt_x2, pr_x2)
    inter_y2 = np.minimum(gt_y2, pr_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_gt = np.maximum(0.0, gt_x2 - gt_x1) * np.maximum(0.0, gt_y2 - gt_y1)
    area_pr = np.maximum(0.0, pr_x2 - pr_x1) * np.maximum(0.0, pr_y2 - pr_y1)

    union = area_gt + area_pr - inter + 1e-12
    return inter / union


def compute_ap_101(recalls, precisions):
    """
    AP interpolée sur 101 points (style COCO/VOC).
    """
    if len(recalls) == 0:
        return 0.0

    recalls = np.asarray(recalls, dtype=np.float32)
    precisions = np.asarray(precisions, dtype=np.float32)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_points:
        idx = np.where(mrec >= r)[0]
        p = np.max(mpre[idx]) if idx.size > 0 else 0.0
        ap += p

    return float(ap / 101.0)


# PARSING GT
def build_video_to_meta_map(videos_root: Path, activity_name: str):
    video_files = sorted(glob.glob(str(videos_root / f"{activity_name}.*.mp4")))
    if not video_files:
        video_files = sorted(glob.glob(str(videos_root / f"{activity_name}.*.avi")))

    mapping = {}
    for cam_id, video_path in enumerate(video_files):
        stem = Path(video_path).stem

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Impossible d'ouvrir la vidéo: {video_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        mapping[stem] = {
            "cam_id": cam_id,
            "width": width,
            "height": height,
            "video_path": str(video_path),
        }

    return mapping


def video_stem_to_pred_prefix(video_stem: str) -> str:
    """
    Ex:
      CMJump_01.51433950.20251218140238
    -> CMJump_01_51433950
    """
    parts = video_stem.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return video_stem.replace(".", "_")


def find_prediction_file(pred_root: Path, activity_name: str, cam_id: int, video_stem: str):
    """
    Cherche le fichier de prédictions
    """
    cam_suffix = f"cam{cam_id:02d}_detections.csv"
    pred_prefix = video_stem_to_pred_prefix(video_stem)

    direct_candidates = [
        pred_root / f"{pred_prefix}_{cam_suffix}",
        pred_root / f"{activity_name}_{cam_suffix}",
        pred_root / f"{video_stem}_{cam_suffix}",
    ]

    for p in direct_candidates:
        if p.exists():
            return p

    # recherche plus souple
    token = None
    parts = video_stem.split(".")
    if len(parts) >= 2:
        token = parts[1]

    patterns = [
        f"*{pred_prefix}*{cam_suffix}",
        f"*{activity_name}*{cam_suffix}",
        f"*{token}*{cam_suffix}" if token else None,
        f"*{cam_suffix}",
    ]

    matches = []
    for pat in patterns:
        if pat is None:
            continue
        matches.extend(glob.glob(str(pred_root / pat)))

    matches = sorted(set(matches))

    if len(matches) == 1:
        return Path(matches[0])

    if len(matches) > 1:
        print(f"[WARN] Plusieurs fichiers candidats pour cam{cam_id:02d} / {video_stem}:")
        for m in matches:
            print(f"       - {m}")
        return Path(matches[0])

    return None

def find_prediction_file_for_cam(activity_name: str, cam_id: int):
    pattern = f"{activity_name}_*_cam{cam_id:02d}_detections.csv"
    candidates = sorted(PRED_ROOT.glob(pattern))

    if len(candidates) == 0:
        print(f"[WARN] Aucun fichier prédiction pour {activity_name} cam{cam_id:02d}")
        return None

    if len(candidates) > 1:
        print(f"[WARN] Plusieurs fichiers trouvés pour {activity_name} cam{cam_id:02d}:")
        for c in candidates:
            print(f"       - {c}")
        print(f"[WARN] Je prends le premier: {candidates[0].name}")

    return candidates[0]

def parse_gt_filename(stem: str, activity_name: str):
    prefix = activity_name + "__"
    if not stem.startswith(prefix):
        raise ValueError(f"Nom inattendu: {stem}")

    remainder = stem[len(prefix):]
    if "__f" not in remainder:
        raise ValueError(f"Impossible de parser la frame dans: {stem}")

    video_stem, frame_str = remainder.rsplit("__f", 1)
    frame_idx_gt = int(frame_str)
    return video_stem, frame_idx_gt


def load_yolo_label_file(txt_path: Path, img_w: int, img_h: int):
    boxes = []
    if not txt_path.exists():
        return boxes

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            cls_id, cx, cy, w, h = parts
            cls_id = int(float(cls_id))
            if cls_id != CLASS_ID:
                continue

            cx, cy, w, h = map(float, [cx, cy, w, h])
            boxes.append({
                "bbox": yolo_to_xyxy(cx, cy, w, h, img_w, img_h),
                "class_id": cls_id,
            })

    return boxes


def load_gt_activity(activity_dir: Path):
    """
    On lit val.txt pour connaître exactement les frames GT à évaluer.
    """
    activity_name = activity_dir.name
    val_txt = activity_dir / f"{SUBSET}.txt"
    labels_dir = activity_dir / "labels" / SUBSET

    if not val_txt.exists():
        raise FileNotFoundError(f"val.txt introuvable: {val_txt}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels introuvables: {labels_dir}")

    video_to_meta = build_video_to_meta_map(VIDEOS_ROOT, activity_name)
    if not video_to_meta:
        print(f"[WARN] Aucune vidéo trouvée pour {activity_name}")
        return {}

    gt_by_key = {}

    with open(val_txt, "r", encoding="utf-8") as f:
        rel_image_paths = [line.strip() for line in f if line.strip()]

    for rel_img_path in rel_image_paths:
        img_name = Path(rel_img_path).name
        stem = Path(img_name).stem

        try:
            video_stem, frame_idx_gt = parse_gt_filename(stem, activity_name)
        except Exception as e:
            print(f"[WARN] {e}")
            continue

        if video_stem not in video_to_meta:
            print(f"[WARN] video_stem absent du mapping: {video_stem}")
            continue

        cam_id = video_to_meta[video_stem]["cam_id"]
        img_w = video_to_meta[video_stem]["width"]
        img_h = video_to_meta[video_stem]["height"]

        frame_eval = frame_idx_gt + FRAME_OFFSET

        txt_path = labels_dir / f"{stem}.txt"
        boxes = load_yolo_label_file(txt_path, img_w, img_h)

        key = (activity_name, cam_id, frame_eval)
        gt_by_key[key] = {
            "boxes": boxes,
            "label_path": str(txt_path),
            "img_w": img_w,
            "img_h": img_h,
            "video_stem": video_stem,
            "frame_idx_gt": frame_idx_gt,
            "frame_eval": frame_eval,
            "rel_image_path": rel_img_path,
        }

    return gt_by_key


# PARSING PREDICTIONS RAW DETECTION
def load_predictions_raw_for_activity(activity_name: str, gt_by_key: dict):
    pred_by_key = defaultdict(list)

    valid_pairs = {(cam_id, frame_id) for (_, cam_id, frame_id) in gt_by_key.keys()}
    cam_ids = sorted({cam_id for (_, cam_id, _) in gt_by_key.keys()})

    total_kept = 0

    for cam_id in cam_ids:
        pred_file = find_prediction_file_for_cam(activity_name, cam_id)

        if pred_file is None:
            continue

        print(f"[DEBUG] cam{cam_id:02d} -> fichier prédiction: {pred_file.name}")

        line_count = 0
        kept_count = 0

        with open(pred_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                line_count += 1
                try:
                    frame = int(row["frame"])
                    camera_id = int(row["camera_id"])

                    if camera_id != cam_id:
                        continue

                    if (camera_id, frame) not in valid_pairs:
                        continue

                    score = float(row["score"])
                    if SCORE_THRESHOLD is not None and score < SCORE_THRESHOLD:
                        continue

                    class_id = int(float(row.get("class_id", 0)))
                    if class_id != CLASS_ID:
                        continue

                    bbox = [
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["x2"]),
                        float(row["y2"]),
                    ]

                    key = (activity_name, camera_id, frame)
                    pred_by_key[key].append({
                        "bbox": bbox,
                        "score": score,
                        "class_id": class_id,
                    })
                    kept_count += 1

                except Exception as e:
                    print(f"[WARN] Ligne ignorée dans {pred_file.name}: {e}")

        total_kept += kept_count
        print(f"[DEBUG] cam{cam_id:02d}: lignes CSV={line_count}, prédictions gardées={kept_count}")

    print(f"[DEBUG] Total prédictions gardées pour {activity_name}: {total_kept}")
    return pred_by_key


# METRIQUES POINT DE FONCTIONNEMENT
def match_boxes_hungarian(gt_boxes, pred_boxes, iou_thr=0.5):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], []

    gt_xyxy = [g["bbox"] for g in gt_boxes]
    pr_xyxy = [p["bbox"] for p in pred_boxes]

    ious = box_iou_matrix(gt_xyxy, pr_xyxy)
    cost = 1.0 - ious

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_ious = []
    for r, c in zip(row_ind, col_ind):
        if ious[r, c] >= iou_thr:
            matches.append((r, c))
            matched_ious.append(float(ious[r, c]))

    return matches, matched_ious


def compute_operating_metrics(gt_by_key, pred_by_key, iou_thr=0.5):
    tp = 0
    fp = 0
    fn = 0
    n_gt = 0
    n_pred = 0
    matched_ious = []

    for key in sorted(gt_by_key.keys()):
        gt_boxes = gt_by_key[key]["boxes"]
        pred_boxes = pred_by_key.get(key, [])

        n_gt += len(gt_boxes)
        n_pred += len(pred_boxes)

        matches, ious = match_boxes_hungarian(gt_boxes, pred_boxes, iou_thr=iou_thr)
        m = len(matches)

        tp += m
        fp += max(0, len(pred_boxes) - m)
        fn += max(0, len(gt_boxes) - m)
        matched_ious.extend(ious)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "n_images": len(gt_by_key),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
    }


# AP / mAP
def collect_predictions_sorted(gt_by_key, pred_by_key):
    preds = []
    total_gt = 0

    for key, gt_info in gt_by_key.items():
        total_gt += len(gt_info["boxes"])
        for pred in pred_by_key.get(key, []):
            preds.append({
                "key": key,
                "score": float(pred["score"]),
                "bbox": pred["bbox"],
            })

    preds.sort(key=lambda x: x["score"], reverse=True)
    return preds, total_gt


def compute_ap_at_iou(gt_by_key, pred_by_key, iou_thr=0.5):
    preds, total_gt = collect_predictions_sorted(gt_by_key, pred_by_key)

    if total_gt == 0:
        return 0.0, [], []

    gt_used = {}
    for key, gt_info in gt_by_key.items():
        gt_used[key] = np.zeros(len(gt_info["boxes"]), dtype=bool)

    tp = []
    fp = []

    for pred in preds:
        key = pred["key"]
        pred_box = pred["bbox"]
        gt_boxes = gt_by_key[key]["boxes"]

        if len(gt_boxes) == 0:
            tp.append(0)
            fp.append(1)
            continue

        best_iou = -1.0
        best_gt_idx = -1

        for i, gt in enumerate(gt_boxes):
            if gt_used[key][i]:
                continue
            iou = compute_iou(gt["bbox"], pred_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thr and best_gt_idx >= 0:
            gt_used[key][best_gt_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recalls = tp / max(total_gt, 1)
    precisions = tp / np.maximum(tp + fp, 1e-12)

    ap = compute_ap_101(recalls, precisions)
    return ap, recalls.tolist(), precisions.tolist()


def compute_map_metrics(gt_by_key, pred_by_key):
    ap_by_iou = {}

    for iou_thr in MAP_IOUS:
        ap, _, _ = compute_ap_at_iou(gt_by_key, pred_by_key, iou_thr=float(iou_thr))
        ap_by_iou[round(float(iou_thr), 2)] = ap

    ap50 = ap_by_iou.get(0.50, 0.0)
    map50_95 = float(np.mean(list(ap_by_iou.values()))) if ap_by_iou else 0.0

    return {
        "ap50": ap50,
        "map50_95": map50_95,
        "ap_by_iou": ap_by_iou,
    }

def video_stem_to_pred_prefix(video_stem: str) -> str:
    parts = video_stem.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return video_stem.replace(".", "_")

# MERGE GLOBAL
def merge_dict_of_images(dicts):
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged

# MAIN
def main():
    ensure_dir(OUTPUT_DIR)

    if ACTIVITIES is None:
        activity_dirs = sorted([p for p in GT_ROOT.iterdir() if p.is_dir()])
    else:
        activity_dirs = [GT_ROOT / a for a in ACTIVITIES if (GT_ROOT / a).is_dir()]

    if not activity_dirs:
        print("[ERREUR] Aucune activité trouvée.")
        return

    all_gt_dicts = []
    all_pred_dicts = []
    summary_rows = []

    print("=" * 80)
    print("ÉVALUATION DÉTECTION")
    print(f"GT_ROOT         : {GT_ROOT}")
    print(f"VIDEOS_ROOT     : {VIDEOS_ROOT}")
    print(f"PRED_ROOT       : {PRED_ROOT}")
    print(f"OUTPUT_DIR      : {OUTPUT_DIR}")
    print(f"SCORE_THRESHOLD : {SCORE_THRESHOLD}")
    print(f"OPERATING_IOU   : {OPERATING_IOU}")
    print("=" * 80)

    for activity_dir in activity_dirs:
        activity_name = activity_dir.name
        print(f"\n[INFO] Activité: {activity_name}")

        gt_by_key = load_gt_activity(activity_dir)
        if not gt_by_key:
            print(f"[WARN] GT vide pour {activity_name}")
            continue

        pred_by_key = load_predictions_raw_for_activity(activity_name, gt_by_key)

        op_metrics = compute_operating_metrics(
            gt_by_key=gt_by_key,
            pred_by_key=pred_by_key,
            iou_thr=OPERATING_IOU
        )

        map_metrics = compute_map_metrics(
            gt_by_key=gt_by_key,
            pred_by_key=pred_by_key
        )

        row = {
            "activity": activity_name,
            "n_images": op_metrics["n_images"],
            "n_gt": op_metrics["n_gt"],
            "n_pred": op_metrics["n_pred"],
            "tp": op_metrics["tp"],
            "fp": op_metrics["fp"],
            "fn": op_metrics["fn"],
            "precision": op_metrics["precision"],
            "recall": op_metrics["recall"],
            "f1": op_metrics["f1"],
            "mean_iou": op_metrics["mean_iou"],
            "ap50": map_metrics["ap50"],
            "map50_95": map_metrics["map50_95"],
        }
        summary_rows.append(row)

        print(
            f"  images={row['n_images']} | gt={row['n_gt']} | pred={row['n_pred']} | "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']}"
        )
        print(
            f"  Precision={row['precision']:.4f} | Recall={row['recall']:.4f} | "
            f"F1={row['f1']:.4f} | IoU_moy={row['mean_iou']:.4f}"
        )
        print(
            f"  AP50={row['ap50']:.4f} | mAP50-95={row['map50_95']:.4f}"
        )

        ap_iou_csv = OUTPUT_DIR / f"{activity_name}_ap_by_iou.csv"
        with open(ap_iou_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["iou", "ap"])
            for iou_thr, ap_val in map_metrics["ap_by_iou"].items():
                writer.writerow([iou_thr, ap_val])

        all_gt_dicts.append(gt_by_key)
        all_pred_dicts.append(pred_by_key)

    # GLOBAL
    if summary_rows:
        global_gt = merge_dict_of_images(all_gt_dicts)
        global_pred = merge_dict_of_images(all_pred_dicts)

        global_op = compute_operating_metrics(
            gt_by_key=global_gt,
            pred_by_key=global_pred,
            iou_thr=OPERATING_IOU
        )

        global_map = compute_map_metrics(
            gt_by_key=global_gt,
            pred_by_key=global_pred
        )

        print("\n" + "=" * 80)
        print("RÉSULTATS GLOBAUX")
        print(
            f"images={global_op['n_images']} | gt={global_op['n_gt']} | pred={global_op['n_pred']} | "
            f"TP={global_op['tp']} FP={global_op['fp']} FN={global_op['fn']}"
        )
        print(
            f"Precision={global_op['precision']:.4f} | Recall={global_op['recall']:.4f} | "
            f"F1={global_op['f1']:.4f} | IoU_moy={global_op['mean_iou']:.4f}"
        )
        print(
            f"AP50={global_map['ap50']:.4f} | mAP50-95={global_map['map50_95']:.4f}"
        )
        print("=" * 80)

        summary_csv = OUTPUT_DIR / "summary_per_activity.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "activity", "n_images", "n_gt", "n_pred",
                "tp", "fp", "fn",
                "precision", "recall", "f1", "mean_iou",
                "ap50", "map50_95"
            ])
            for row in summary_rows:
                writer.writerow([
                    row["activity"],
                    row["n_images"],
                    row["n_gt"],
                    row["n_pred"],
                    row["tp"],
                    row["fp"],
                    row["fn"],
                    row["precision"],
                    row["recall"],
                    row["f1"],
                    row["mean_iou"],
                    row["ap50"],
                    row["map50_95"],
                ])

        global_ap_csv = OUTPUT_DIR / "global_ap_by_iou.csv"
        with open(global_ap_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["iou", "ap"])
            for iou_thr, ap_val in global_map["ap_by_iou"].items():
                writer.writerow([iou_thr, ap_val])

        global_txt = OUTPUT_DIR / "global_results.txt"
        with open(global_txt, "w", encoding="utf-8") as f:
            f.write("EVALUATION DETECTION\n")
            f.write("=" * 60 + "\n")
            f.write(f"GT_ROOT: {GT_ROOT}\n")
            f.write(f"VIDEOS_ROOT: {VIDEOS_ROOT}\n")
            f.write(f"PRED_ROOT: {PRED_ROOT}\n")
            f.write(f"SCORE_THRESHOLD: {SCORE_THRESHOLD}\n")
            f.write(f"OPERATING_IOU: {OPERATING_IOU}\n")
            f.write("\n")
            f.write(f"n_images: {global_op['n_images']}\n")
            f.write(f"n_gt: {global_op['n_gt']}\n")
            f.write(f"n_pred: {global_op['n_pred']}\n")
            f.write(f"TP: {global_op['tp']}\n")
            f.write(f"FP: {global_op['fp']}\n")
            f.write(f"FN: {global_op['fn']}\n")
            f.write(f"Precision: {global_op['precision']:.6f}\n")
            f.write(f"Recall: {global_op['recall']:.6f}\n")
            f.write(f"F1: {global_op['f1']:.6f}\n")
            f.write(f"Mean IoU: {global_op['mean_iou']:.6f}\n")
            f.write(f"AP50: {global_map['ap50']:.6f}\n")
            f.write(f"mAP50-95: {global_map['map50_95']:.6f}\n")

        print(f"[OK] Résultats sauvés dans: {OUTPUT_DIR}")

    else:
        print("[WARN] Aucun résultat calculé.")


if __name__ == "__main__":
    main()