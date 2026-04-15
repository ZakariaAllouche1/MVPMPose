#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import motmetrics as mm
except ImportError as exc:
    raise SystemExit(
        "Le package 'motmetrics' est requis.\n"
        "Installe-le avec : pip install motmetrics\n"
        f"Erreur: {exc}"
    )



# CONFIG
# GT attendus sous forme de dossiers:
#   GT_Tracking_CMJump01_s12_cam00/
#   GT_Tracking_CMJump01_s12_cam01/
# avec dedans :
#   gt/gt.txt
#
# Prédictions attendues :
#   CMJump_01_51433950_cam00_tracking_mot.txt
#   CMJump_01_51433950_cam01_tracking_mot.txt
#   ...

GT_ROOT = Path("./GT_track_metasport12")
PRED_ROOT = Path("./results_1/videos_eval/ResultsMETASPORT12Rtmlib_V3")
OUTPUT_DIR = Path("./eval_tracking_subject12_ancien")

ACTIVITIES = None       
GT_SUBJECT_TAG = "s12"

IOU_THRESHOLD = 0.50

EVAL_TRUE_GLOBAL_IF_AVAILABLE = False


#utils
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_int(x):
    return int(float(x))


def safe_float(x):
    return float(x)


def strip_prediction_suffix(activity_name: str) -> str:
    """
    Retire le suffixe final numérique ajouté dans les prédictions.
    Exemple :
      CMJump_01_51433950 -> CMJump_01
    """
    return re.sub(r"_\d+$", "", activity_name)


def activity_to_gt_token(activity_name: str) -> str:
    """
    Convertit le nom activité côté prédiction
    Exemple :
      CMJump_01_51433950 -> CMJump01
    """
    base_activity = strip_prediction_suffix(activity_name)
    return base_activity.replace("_", "")


def discover_activities():
    """
    Détecte automatiquement les activités depuis les fichiers de prédiction :
      CMJump_01_51433950_cam00_tracking_mot.txt -> activité = CMJump_01_51433950
    """
    if ACTIVITIES is not None:
        return list(ACTIVITIES)

    activities = set()
    pattern = re.compile(r"^(?P<activity>.+)_cam\d{2}_tracking_mot\.txt$", re.IGNORECASE)

    for path in PRED_ROOT.glob("*_cam*_tracking_mot.txt"):
        m = pattern.match(path.name)
        if m:
            activities.add(m.group("activity"))

    return sorted(activities)


def compute_iou_xywh(box_a, box_b):
    ax, ay, aw, ah = map(float, box_a)
    bx, by, bw, bh = map(float, box_b)

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)

    union = area_a + area_b - inter + 1e-12
    return inter / union


def build_iou_distance_matrix(gt_boxes, pred_boxes, iou_thr=0.5):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.empty((len(gt_boxes), len(pred_boxes)))

    dists = np.full((len(gt_boxes), len(pred_boxes)), np.nan, dtype=float)

    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            iou = compute_iou_xywh(g, p)
            if iou >= iou_thr:
                dists[i, j] = 1.0 - iou

    return dists


def rename_summary_columns(df):
    return df.rename(columns={
        "num_frames": "n_frames",
        "num_objects": "n_gt",
        "num_predictions": "n_pred",
        "num_matches": "matches",
        "num_switches": "IDs",
        "num_fragmentations": "FM",
        "mostly_tracked": "MT",
        "partially_tracked": "PT",
        "mostly_lost": "ML",
        "num_false_positives": "FP",
        "num_misses": "FN",
        "recall": "recall",
        "precision": "precision",
        "mota": "mota",
        "motp": "motp",
        "idf1": "idf1",
        "idp": "idp",
        "idr": "idr",
    })


def parse_mot_file(txt_path: Path):
    """
    Format MOT attendu:
    frame,id,x,y,w,h,conf,-1,-1,-1
    """
    data = defaultdict(list)

    if not txt_path.exists():
        return data

    with open(txt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, start=1):
            if not row:
                continue

            if len(row) < 6:
                raise ValueError(f"Ligne invalide dans {txt_path} à la ligne {line_num}: {row}")

            frame = safe_int(row[0])
            track_id = safe_int(row[1])
            x = safe_float(row[2])
            y = safe_float(row[3])
            w = safe_float(row[4])
            h = safe_float(row[5])

            if w <= 0 or h <= 0:
                continue

            data[frame].append({
                "id": track_id,
                "bbox": [x, y, w, h],
            })

    return data


def build_accumulator(gt_frames, pred_frames, iou_thr=0.5):
    acc = mm.MOTAccumulator(auto_id=True)

    frame_ids = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))

    for frame_id in frame_ids:
        gt_list = gt_frames.get(frame_id, [])
        pred_list = pred_frames.get(frame_id, [])

        gt_ids = [g["id"] for g in gt_list]
        pred_ids = [p["id"] for p in pred_list]

        gt_boxes = [g["bbox"] for g in gt_list]
        pred_boxes = [p["bbox"] for p in pred_list]

        dist_matrix = build_iou_distance_matrix(gt_boxes, pred_boxes, iou_thr=iou_thr)

        # NE PAS passer frameid quand auto_id=True
        acc.update(gt_ids, pred_ids, dist_matrix)

    return acc


def evaluate_accumulators(accumulators_dict):
    if not accumulators_dict:
        return None

    mh = mm.metrics.create()

    metric_names = [
        "num_frames",
        "num_objects",
        "num_predictions",
        "num_matches",
        "num_switches",
        "num_fragmentations",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "num_false_positives",
        "num_misses",
        "precision",
        "recall",
        "mota",
        "motp",
        "idf1",
        "idp",
        "idr",
    ]

    names = list(accumulators_dict.keys())
    accs = list(accumulators_dict.values())

    summary = mh.compute_many(
        accs,
        names=names,
        metrics=metric_names,
        generate_overall=True,
    )

    summary = rename_summary_columns(summary)
    return summary


def resolve_gt_txt_from_dir(src: Path):
    """
    Résout le fichier GT à partir d'un dossier GT.

    Cas supportés :
      dossier/gt.txt
      dossier/gt/gt.txt
      dossier/<unique .txt>
      dossier/gt/<unique .txt>
    """
    candidates = [
        src / "gt.txt",
        src / "gt" / "gt.txt",
    ]

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    txts = sorted([p for p in src.glob("*.txt") if p.is_file()])
    if len(txts) == 1:
        return txts[0]

    gt_dir = src / "gt"
    if gt_dir.exists() and gt_dir.is_dir():
        txts_in_gt = sorted([p for p in gt_dir.glob("*.txt") if p.is_file()])
        if len(txts_in_gt) == 1:
            return txts_in_gt[0]

    return None

# Parsing GT / PRED
def load_gt_activity(activity_name: str):
    """
    Cherche des DOSSIERS GT du style :
      GT_Tracking_CMJump01_s12_cam00/
      GT_Tracking_CMJump01_s12_cam01/
    avec gt/gt.txt dedans.
    """
    gt_token = activity_to_gt_token(activity_name)

    cam_dir_pattern = f"GT_Tracking_{gt_token}_{GT_SUBJECT_TAG}_cam*"
    gt_cam_dirs = sorted([p for p in GT_ROOT.rglob(cam_dir_pattern) if p.is_dir()])

    gt_by_cam = {}
    gt_cam_sources = []

    cam_re = re.compile(r"_cam(\d+)$", re.IGNORECASE)

    for gt_dir in gt_cam_dirs:
        m = cam_re.search(gt_dir.name)
        if not m:
            print(f"[WARN] Impossible d'extraire le cam_id depuis le dossier {gt_dir.name}")
            continue

        cam_id = int(m.group(1))
        gt_txt = resolve_gt_txt_from_dir(gt_dir)

        if gt_txt is None:
            print(f"[WARN] Aucun fichier GT exploitable trouvé dans {gt_dir}")
            continue

        gt_by_cam[cam_id] = parse_mot_file(gt_txt)
        gt_cam_sources.append((cam_id, gt_dir, gt_txt))

    gt_global = None
    global_dir_pattern = f"GT_Tracking_{gt_token}_{GT_SUBJECT_TAG}_global"
    gt_global_dirs = sorted([p for p in GT_ROOT.rglob(global_dir_pattern) if p.is_dir()])
    if gt_global_dirs:
        gt_global_txt = resolve_gt_txt_from_dir(gt_global_dirs[0])
        if gt_global_txt is not None:
            gt_global = parse_mot_file(gt_global_txt)

    return {
        "activity": activity_name,
        "gt_token": gt_token,
        "per_cam": gt_by_cam,
        "global": gt_global,
        "gt_cam_sources": gt_cam_sources,
    }


def find_prediction_file_for_cam(activity_name: str, cam_id: int):
    pattern = f"{activity_name}_cam{cam_id:02d}_tracking_mot.txt"
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


def find_prediction_file_global(activity_name: str):
    pattern = f"{activity_name}_global_tracking_mot.txt"
    candidates = sorted(PRED_ROOT.glob(pattern))

    if len(candidates) == 0:
        print(f"[WARN] Aucun fichier prédiction globale pour {activity_name}")
        return None

    if len(candidates) > 1:
        print(f"[WARN] Plusieurs fichiers globaux trouvés pour {activity_name}:")
        for c in candidates:
            print(f"       - {c}")
        print(f"[WARN] Je prends le premier: {candidates[0].name}")

    return candidates[0]


def load_predictions_activity(activity_name: str, gt_info: dict):
    pred_by_cam = {}

    for cam_id in sorted(gt_info["per_cam"].keys()):
        pred_file = find_prediction_file_for_cam(activity_name, cam_id)
        if pred_file is None:
            continue
        pred_by_cam[cam_id] = parse_mot_file(pred_file)

    pred_global = None
    if EVAL_TRUE_GLOBAL_IF_AVAILABLE and gt_info["global"] is not None:
        pred_global_file = find_prediction_file_global(activity_name)
        if pred_global_file is not None:
            pred_global = parse_mot_file(pred_global_file)

    return {
        "per_cam": pred_by_cam,
        "global": pred_global,
    }


#main
def main():
    ensure_dir(OUTPUT_DIR)

    activities = discover_activities()

    if not activities:
        print("[ERREUR] Aucune activité trouvée.")
        return

    summary_rows = []
    all_accumulators = {}
    global_true_accumulators = {}

    print("=" * 80)
    print("EVALUATION TRACKING")
    print(f"GT_ROOT                         : {GT_ROOT}")
    print(f"PRED_ROOT                       : {PRED_ROOT}")
    print(f"OUTPUT_DIR                      : {OUTPUT_DIR}")
    print(f"IOU_THRESHOLD                   : {IOU_THRESHOLD}")
    print(f"GT_SUBJECT_TAG                  : {GT_SUBJECT_TAG}")
    print(f"EVAL_TRUE_GLOBAL_IF_AVAILABLE   : {EVAL_TRUE_GLOBAL_IF_AVAILABLE}")
    print(f"ACTIVITIES                      : {activities}")
    print("=" * 80)

    for activity_name in activities:
        print(f"\n[INFO] Activité: {activity_name}")

        gt_info = load_gt_activity(activity_name)

        if not gt_info["per_cam"]:
            expected = f"GT_Tracking_{gt_info['gt_token']}_{GT_SUBJECT_TAG}_camXX/"
            print(f"[WARN] Aucun GT caméra pour {activity_name}")
            print(f"       Base activité prédiction : {strip_prediction_suffix(activity_name)}")
            print(f"       Token GT attendu         : {gt_info['gt_token']}")
            print(f"       Pattern attendu          : {expected}")
            print("       Avec gt/gt.txt dedans.")
            continue

        print("  GT trouvés:")
        for cam_id, gt_dir, gt_txt in gt_info["gt_cam_sources"]:
            print(f"    cam{cam_id:02d} -> dossier={gt_dir.name} | fichier={gt_txt}")

        pred_info = load_predictions_activity(activity_name, gt_info)
        activity_accumulators = {}

        for cam_id, gt_frames in sorted(gt_info["per_cam"].items()):
            pred_frames = pred_info["per_cam"].get(cam_id, defaultdict(list))
            seq_name = f"{activity_name}_cam{cam_id:02d}"

            acc = build_accumulator(
                gt_frames=gt_frames,
                pred_frames=pred_frames,
                iou_thr=IOU_THRESHOLD
            )

            activity_accumulators[seq_name] = acc
            all_accumulators[seq_name] = acc

            n_gt_frames = len(gt_frames)
            n_pred_frames = len(pred_frames)
            print(f"  cam{cam_id:02d} | gt_frames={n_gt_frames} | pred_frames={n_pred_frames}")

        activity_summary = evaluate_accumulators(activity_accumulators)
        if activity_summary is None:
            print(f"[WARN] Aucun accumulateur pour {activity_name}")
            continue

        activity_csv = OUTPUT_DIR / f"{activity_name}_per_camera_metrics.csv"
        activity_summary.to_csv(activity_csv, float_format="%.6f")

        overall_row = activity_summary.loc["OVERALL"]

        row = {
            "activity": activity_name,
            "n_frames": overall_row["n_frames"],
            "n_gt": overall_row["n_gt"],
            "n_pred": overall_row["n_pred"],
            "matches": overall_row["matches"],
            "FP": overall_row["FP"],
            "FN": overall_row["FN"],
            "IDs": overall_row["IDs"],
            "FM": overall_row["FM"],
            "precision": overall_row["precision"],
            "recall": overall_row["recall"],
            "mota": overall_row["mota"],
            "motp": overall_row["motp"],
            "idf1": overall_row["idf1"],
            "idp": overall_row["idp"],
            "idr": overall_row["idr"],
            "MT": overall_row["MT"],
            "PT": overall_row["PT"],
            "ML": overall_row["ML"],
        }
        summary_rows.append(row)

        print(
            f"  OVERALL cameras | GT={row['n_gt']} | PRED={row['n_pred']} | "
            f"FP={row['FP']} | FN={row['FN']} | IDs={row['IDs']}"
        )
        print(
            f"  Precision={row['precision']:.4f} | Recall={row['recall']:.4f} | "
            f"MOTA={row['mota']:.4f} | IDF1={row['idf1']:.4f}"
        )

        if EVAL_TRUE_GLOBAL_IF_AVAILABLE and gt_info["global"] is not None and pred_info["global"] is not None:
            global_seq_name = f"{activity_name}_global"
            global_true_accumulators[global_seq_name] = build_accumulator(
                gt_frames=gt_info["global"],
                pred_frames=pred_info["global"],
                iou_thr=IOU_THRESHOLD
            )

    if summary_rows:
        summary_csv = OUTPUT_DIR / "summary_per_activity.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "activity", "n_frames", "n_gt", "n_pred", "matches",
                "FP", "FN", "IDs", "FM",
                "precision", "recall", "mota", "motp",
                "idf1", "idp", "idr", "MT", "PT", "ML"
            ])
            for row in summary_rows:
                writer.writerow([
                    row["activity"],
                    row["n_frames"],
                    row["n_gt"],
                    row["n_pred"],
                    row["matches"],
                    row["FP"],
                    row["FN"],
                    row["IDs"],
                    row["FM"],
                    row["precision"],
                    row["recall"],
                    row["mota"],
                    row["motp"],
                    row["idf1"],
                    row["idp"],
                    row["idr"],
                    row["MT"],
                    row["PT"],
                    row["ML"],
                ])

    if all_accumulators:
        global_summary = evaluate_accumulators(all_accumulators)

        print("\n" + "=" * 80)
        print("RESULTATS GLOBAUX AGREGES (sur toutes les séquences caméra)")
        print("=" * 80)

        overall_row = global_summary.loc["OVERALL"]
        print(
            f"frames={overall_row['n_frames']} | gt={overall_row['n_gt']} | pred={overall_row['n_pred']} | "
            f"FP={overall_row['FP']} | FN={overall_row['FN']} | IDs={overall_row['IDs']}"
        )
        print(
            f"Precision={overall_row['precision']:.4f} | Recall={overall_row['recall']:.4f} | "
            f"MOTA={overall_row['mota']:.4f} | MOTP={overall_row['motp']:.4f} | "
            f"IDF1={overall_row['idf1']:.4f} | IDP={overall_row['idp']:.4f} | IDR={overall_row['idr']:.4f}"
        )
        print("=" * 80)

        global_summary_csv = OUTPUT_DIR / "summary_per_sequence.csv"
        global_summary.to_csv(global_summary_csv, float_format="%.6f")

        global_txt = OUTPUT_DIR / "global_results.txt"
        with open(global_txt, "w", encoding="utf-8") as f:
            f.write("EVALUATION TRACKING\n")
            f.write("=" * 60 + "\n")
            f.write(f"GT_ROOT: {GT_ROOT}\n")
            f.write(f"PRED_ROOT: {PRED_ROOT}\n")
            f.write(f"IOU_THRESHOLD: {IOU_THRESHOLD}\n")
            f.write(f"GT_SUBJECT_TAG: {GT_SUBJECT_TAG}\n")
            f.write("\n")
            f.write("GLOBAL AGREGE SUR LES SEQUENCES CAMERA\n")
            f.write(f"n_frames: {overall_row['n_frames']}\n")
            f.write(f"n_gt: {overall_row['n_gt']}\n")
            f.write(f"n_pred: {overall_row['n_pred']}\n")
            f.write(f"matches: {overall_row['matches']}\n")
            f.write(f"FP: {overall_row['FP']}\n")
            f.write(f"FN: {overall_row['FN']}\n")
            f.write(f"IDs: {overall_row['IDs']}\n")
            f.write(f"FM: {overall_row['FM']}\n")
            f.write(f"Precision: {overall_row['precision']:.6f}\n")
            f.write(f"Recall: {overall_row['recall']:.6f}\n")
            f.write(f"MOTA: {overall_row['mota']:.6f}\n")
            f.write(f"MOTP: {overall_row['motp']:.6f}\n")
            f.write(f"IDF1: {overall_row['idf1']:.6f}\n")
            f.write(f"IDP: {overall_row['idp']:.6f}\n")
            f.write(f"IDR: {overall_row['idr']:.6f}\n")
            f.write(f"MT: {overall_row['MT']}\n")
            f.write(f"PT: {overall_row['PT']}\n")
            f.write(f"ML: {overall_row['ML']}\n")

        print(f"[OK] Résultats sauvés dans: {OUTPUT_DIR}")

    else:
        print("[WARN] Aucun résultat calculé.")

    if EVAL_TRUE_GLOBAL_IF_AVAILABLE and global_true_accumulators:
        true_global_summary = evaluate_accumulators(global_true_accumulators)
        true_global_csv = OUTPUT_DIR / "true_global_multicam_summary.csv"
        true_global_summary.to_csv(true_global_csv, float_format="%.6f")
        print(f"[OK] Vrai global multi-cam sauvé: {true_global_csv}")


if __name__ == "__main__":
    main()