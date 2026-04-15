from pathlib import Path
import shutil
import zipfile

ROOT = Path("./yolo_dataset_clean_V1")
OUT_ZIP = Path("./cvat_ultralytics_yolo_V1.zip")

splits = {
    "train": ROOT / "train.txt",
    "val": ROOT / "val.txt",
}

tmp = ROOT / "_cvat_ultra_tmp"
if tmp.exists():
    shutil.rmtree(tmp)

for split, txt_file in splits.items():
    img_dst = tmp / "images" / split
    lbl_dst = tmp / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    image_paths = [x.strip() for x in txt_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    for img_path_str in image_paths:
        # Convertir les chemins absolus en chemins relatifs
        if img_path_str.startswith("/"):
            img_name = img_path_str.split("/")[-1]
            img_path = ROOT / "images" / "all" / img_name
        else:
            img_path = Path(img_path_str)
        
        lbl_path = ROOT / "labels" / "all" / f"{img_path.stem}.txt"
        if not img_path.exists() or not lbl_path.exists():
            continue
        shutil.copy2(img_path, img_dst / img_path.name)
        shutil.copy2(lbl_path, lbl_dst / lbl_path.name)

data_yaml = tmp / "data.yaml"
data_yaml.write_text(
    "path: ./\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: person\n",
    encoding="utf-8",
)

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in tmp.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(tmp))

print(f"[OK] ZIP créé: {OUT_ZIP.resolve()}")