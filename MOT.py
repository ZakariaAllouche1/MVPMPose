from pathlib import Path
import zipfile

src_txt = Path("GT_tracking/videos_eval_new/ResultsMETASPORT12Rtmlib_V3/Deadlift_01_51433950_cam03_tracking_mot.txt")
work_dir = Path("mot_import_tmp")
gt_dir = work_dir / "gt"
gt_dir.mkdir(parents=True, exist_ok=True)

src_lines = src_txt.read_text(encoding="utf-8").strip().splitlines()

out_lines = []
for line in src_lines:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6:
        continue

    frame_id = parts[0]
    track_id = parts[1]
    x = parts[2]
    y = parts[3]
    w = parts[4]
    h = parts[5]

    # Format GT MOT attendu par CVAT:
    # frame_id, track_id, x, y, w, h, not_ignored, class_id, visibility
    out_lines.append(f"{frame_id},{track_id},{x},{y},{w},{h},1,1,1")

(gt_dir / "gt.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
(gt_dir / "labels.txt").write_text("person\n", encoding="utf-8")

zip_path = Path("deadlift01_cam03_mot_import.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(gt_dir / "gt.txt", arcname="gt/gt.txt")
    zf.write(gt_dir / "labels.txt", arcname="gt/labels.txt")

print(f"ZIP créé : {zip_path}")