#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import defaultdict
import subprocess
import sys


def group_videos_recursive(folder: Path):
    groups = defaultdict(list)

    for f in folder.rglob("*_tracked_v68.mp4"):
        name = f.stem
        parts = name.split(".")

        # ex:
        # CMJump_01.51433950.20251218140238_tracked_v68
        if len(parts) < 3:
            continue

        prefix = parts[0]      # CMJump_01
        cam_id = parts[1]      # 51433950

        # on groupe par dossier parent + prefix
        key = (f.parent, prefix)
        groups[key].append((cam_id, f))

    return groups


def make_4views(input_files, output_file: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_files[0]),
        "-i", str(input_files[1]),
        "-i", str(input_files[2]),
        "-i", str(input_files[3]),
        "-filter_complex",
        "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0,format=yuv420p[v]",
        "-map", "[v]",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        str(output_file)
    ]

    print(f"\nCréation : {output_file}")
    subprocess.run(cmd, check=True)


def main():
    script_dir = Path(__file__).resolve().parent
    base_results = script_dir / "results"

    print(f"Dossier de base : {base_results}")

    rel_path = Path("videos_V4_finetune/ResultsMETASPORT12Rtmlib_V3")
    folder = base_results / rel_path

    if not folder.exists():
        print(f"Dossier introuvable : {folder}")
        sys.exit(1)

    groups = group_videos_recursive(folder)

    if not groups:
        print(f"Aucune vidéo *_tracked_v68.mp4 trouvée dans : {folder}")
        sys.exit(0)

    for (parent_folder, prefix), items in sorted(groups.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        items = sorted(items, key=lambda x: x[0])

        if len(items) != 4:
            print(f"\n[IGNORÉ] {prefix} dans {parent_folder} : {len(items)} vidéo(s) trouvée(s), il en faut 4.")
            continue

        input_files = [path for _, path in items]
        output_file = parent_folder / f"{prefix}_4views.mp4"

        if output_file.exists():
            print(f"\n[SKIP] {output_file} existe déjà.")
            continue

        try:
            make_4views(input_files, output_file)
        except subprocess.CalledProcessError:
            print(f"[ERREUR] Échec pour {prefix} dans {parent_folder}")

    print("\nTerminé.")


if __name__ == "__main__":
    main()