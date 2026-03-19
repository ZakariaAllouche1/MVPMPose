from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
PRED_CSV = "predictions.csv"
GT_CSV = "ground_truth.csv"
OUTPUT_DIR = "eval_outputs"
IOU_THRESHOLD = 0.5

BOX_COLS = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
REQUIRED_COLS = ["camera_id", "frame_id"] + BOX_COLS


# =========================================================
# LOAD
# =========================================================
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} : colonnes manquantes {missing}")

    out = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "camera_id": df["camera_id"].astype(str),
        "frame_id": df["frame_id"].astype(str),
        "bbox_x1": pd.to_numeric(df["bbox_x1"], errors="coerce"),
        "bbox_y1": pd.to_numeric(df["bbox_y1"], errors="coerce"),
        "bbox_x2": pd.to_numeric(df["bbox_x2"], errors="coerce"),
        "bbox_y2": pd.to_numeric(df["bbox_y2"], errors="coerce"),
    })

    out = out.dropna(subset=BOX_COLS).copy()

    # remettre les coins dans le bon ordre
    x1 = np.minimum(out["bbox_x1"], out["bbox_x2"])
    x2 = np.maximum(out["bbox_x1"], out["bbox_x2"])
    y1 = np.minimum(out["bbox_y1"], out["bbox_y2"])
    y2 = np.maximum(out["bbox_y1"], out["bbox_y2"])

    out["bbox_x1"] = x1
    out["bbox_x2"] = x2
    out["bbox_y1"] = y1
    out["bbox_y2"] = y2

    # enlever les boxes invalides
    out = out[
        (out["bbox_x2"] > out["bbox_x1"]) &
        (out["bbox_y2"] > out["bbox_y1"])
    ].reset_index(drop=True)

    # une image = une caméra + une frame
    out["group_id"] = out["camera_id"] + "__" + out["frame_id"]

    return out


# =========================================================
# IOU
# =========================================================
def compute_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


# =========================================================
# MATCHING
# =========================================================
def greedy_match(pred_img: pd.DataFrame, gt_img: pd.DataFrame, iou_threshold: float):
    """
    Matching glouton global :
    - on calcule toutes les IoU pred/gt
    - on trie par IoU décroissante
    - on garde les couples sans réutiliser une prédiction ou une GT
    """
    candidates = []

    pred_boxes = pred_img[BOX_COLS].to_numpy()
    gt_boxes = gt_img[BOX_COLS].to_numpy()

    for p_idx, p_box in enumerate(pred_boxes):
        for g_idx, g_box in enumerate(gt_boxes):
            iou_value = compute_iou(p_box, g_box)
            if iou_value >= iou_threshold:
                candidates.append((iou_value, p_idx, g_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_preds = set()
    used_gts = set()
    matches = []

    for iou_value, p_idx, g_idx in candidates:
        if p_idx not in used_preds and g_idx not in used_gts:
            used_preds.add(p_idx)
            used_gts.add(g_idx)
            matches.append((p_idx, g_idx, iou_value))

    unmatched_preds = [i for i in range(len(pred_img)) if i not in used_preds]
    unmatched_gts = [i for i in range(len(gt_img)) if i not in used_gts]

    return matches, unmatched_preds, unmatched_gts


# =========================================================
# EVALUATION
# =========================================================
def evaluate_detection(pred_df: pd.DataFrame, gt_df: pd.DataFrame, iou_threshold: float):
    cameras = sorted(set(pred_df["camera_id"]).union(set(gt_df["camera_id"])))
    all_groups = sorted(set(pred_df["group_id"]).union(set(gt_df["group_id"])))

    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious = []

    per_camera_rows = []
    matched_rows = []

    for camera_id in cameras:
        pred_cam = pred_df[pred_df["camera_id"] == camera_id]
        gt_cam = gt_df[gt_df["camera_id"] == camera_id]
        groups = sorted(set(pred_cam["group_id"]).union(set(gt_cam["group_id"])))

        cam_tp = 0
        cam_fp = 0
        cam_fn = 0
        cam_ious = []

        for group_id in groups:
            pred_img = pred_cam[pred_cam["group_id"] == group_id].reset_index(drop=True)
            gt_img = gt_cam[gt_cam["group_id"] == group_id].reset_index(drop=True)

            matches, unmatched_preds, unmatched_gts = greedy_match(pred_img, gt_img, iou_threshold)

            total_tp += len(matches)
            total_fp += len(unmatched_preds)
            total_fn += len(unmatched_gts)

            cam_tp += len(matches)
            cam_fp += len(unmatched_preds)
            cam_fn += len(unmatched_gts)

            for p_idx, g_idx, iou_value in matches:
                pred_row = pred_img.loc[p_idx]
                gt_row = gt_img.loc[g_idx]

                matched_ious.append(iou_value)
                cam_ious.append(iou_value)

                matched_rows.append({
                    "camera_id": camera_id,
                    "frame_id": pred_row["frame_id"],
                    "group_id": group_id,
                    "pred_row_id": int(pred_row["row_id"]),
                    "gt_row_id": int(gt_row["row_id"]),
                    "iou": iou_value,
                    "pred_bbox_x1": pred_row["bbox_x1"],
                    "pred_bbox_y1": pred_row["bbox_y1"],
                    "pred_bbox_x2": pred_row["bbox_x2"],
                    "pred_bbox_y2": pred_row["bbox_y2"],
                    "gt_bbox_x1": gt_row["bbox_x1"],
                    "gt_bbox_y1": gt_row["bbox_y1"],
                    "gt_bbox_x2": gt_row["bbox_x2"],
                    "gt_bbox_y2": gt_row["bbox_y2"],
                })

        cam_precision = cam_tp / (cam_tp + cam_fp) if (cam_tp + cam_fp) > 0 else 0.0
        cam_recall = cam_tp / (cam_tp + cam_fn) if (cam_tp + cam_fn) > 0 else 0.0
        cam_f1 = 2 * cam_precision * cam_recall / (cam_precision + cam_recall) if (cam_precision + cam_recall) > 0 else 0.0
        cam_mean_iou = float(np.mean(cam_ious)) if cam_ious else 0.0

        per_camera_rows.append({
            "camera_id": camera_id,
            "num_frames": len(groups),
            "num_predictions": len(pred_cam),
            "num_ground_truth": len(gt_cam),
            "tp": cam_tp,
            "fp": cam_fp,
            "fn": cam_fn,
            "precision": cam_precision,
            "recall": cam_recall,
            "f1": cam_f1,
            "mean_iou_on_matched_boxes": cam_mean_iou,
        })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    summary_df = pd.DataFrame([{
        "iou_threshold": iou_threshold,
        "num_cameras": len(cameras),
        "num_images_camera_frame": len(all_groups),
        "num_predictions": len(pred_df),
        "num_ground_truth": len(gt_df),
        "num_matches": len(matched_rows),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_on_matched_boxes": mean_iou,
    }])

    per_camera_df = pd.DataFrame(per_camera_rows)
    matches_df = pd.DataFrame(matched_rows)

    return summary_df, per_camera_df, matches_df


# =========================================================
# SAVE CSV
# =========================================================
def save_results(summary_df, per_camera_df, matches_df, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / "evaluation_summary.csv", index=False)
    per_camera_df.to_csv(output_dir / "per_camera_metrics.csv", index=False)
    matches_df.to_csv(output_dir / "matched_boxes.csv", index=False)


# =========================================================
# PLOTS
# =========================================================
def plot_global_counts(summary_df: pd.DataFrame, output_dir: Path):
    row = summary_df.iloc[0]
    labels = ["TP", "FP", "FN"]
    values = [row["tp"], row["fp"], row["fn"]]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("TP / FP / FN globaux")
    plt.ylabel("Nombre")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_global_counts.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_global_metrics(summary_df: pd.DataFrame, output_dir: Path):
    row = summary_df.iloc[0]
    labels = ["Precision", "Recall", "F1", "Mean IoU"]
    values = [row["precision"], row["recall"], row["f1"], row["mean_iou_on_matched_boxes"]]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.title("Métriques globales")
    plt.ylabel("Valeur")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_global_metrics.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_per_camera_counts(per_camera_df: pd.DataFrame, output_dir: Path):
    x = np.arange(len(per_camera_df))
    cameras = per_camera_df["camera_id"].astype(str).tolist()
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, per_camera_df["tp"], width=width, label="TP")
    plt.bar(x, per_camera_df["fp"], width=width, label="FP")
    plt.bar(x + width, per_camera_df["fn"], width=width, label="FN")
    plt.xticks(x, cameras, rotation=20)
    plt.title("TP / FP / FN par caméra")
    plt.ylabel("Nombre")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_per_camera_counts.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_per_camera_metrics(per_camera_df: pd.DataFrame, output_dir: Path):
    x = np.arange(len(per_camera_df))
    cameras = per_camera_df["camera_id"].astype(str).tolist()
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, per_camera_df["precision"], width=width, label="Precision")
    plt.bar(x, per_camera_df["recall"], width=width, label="Recall")
    plt.bar(x + width, per_camera_df["f1"], width=width, label="F1")
    plt.xticks(x, cameras, rotation=20)
    plt.ylim(0, 1)
    plt.title("Precision / Recall / F1 par caméra")
    plt.ylabel("Valeur")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_per_camera_metrics.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_per_camera_mean_iou(per_camera_df: pd.DataFrame, output_dir: Path):
    cameras = per_camera_df["camera_id"].astype(str).tolist()
    values = per_camera_df["mean_iou_on_matched_boxes"].tolist()

    plt.figure(figsize=(8, 4))
    plt.bar(cameras, values)
    plt.ylim(0, 1)
    plt.title("IoU moyen par caméra")
    plt.ylabel("Mean IoU")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_per_camera_mean_iou.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_iou_histogram(matches_df: pd.DataFrame, output_dir: Path):
    if len(matches_df) == 0:
        return

    plt.figure(figsize=(7, 4))
    plt.hist(matches_df["iou"], bins=20)
    plt.title("Distribution des IoU des bbox matchées")
    plt.xlabel("IoU")
    plt.ylabel("Nombre")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_iou_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()


# =========================================================
# MAIN
# =========================================================
def main():
    output_dir = Path(OUTPUT_DIR)

    pred_df = load_csv(PRED_CSV)
    gt_df = load_csv(GT_CSV)

    summary_df, per_camera_df, matches_df = evaluate_detection(
        pred_df=pred_df,
        gt_df=gt_df,
        iou_threshold=IOU_THRESHOLD
    )

    save_results(summary_df, per_camera_df, matches_df, output_dir)

    plot_global_counts(summary_df, output_dir)
    plot_global_metrics(summary_df, output_dir)
    plot_per_camera_counts(per_camera_df, output_dir)
    plot_per_camera_metrics(per_camera_df, output_dir)
    plot_per_camera_mean_iou(per_camera_df, output_dir)
    plot_iou_histogram(matches_df, output_dir)

    print("\n=== Résumé global ===")
    print(summary_df.to_string(index=False))

    print("\n=== Résultats par caméra ===")
    print(per_camera_df.to_string(index=False))

    print(f"\nFichiers sauvegardés dans : {output_dir.resolve()}")


if __name__ == "__main__":
    main()