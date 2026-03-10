import os
import re
import glob
import cv2

OUTPUT_DIR = "output_inference"
FPS = 25

CAMERA_DIRS = ['camera1L', 'camera1R', 'camera2L', 'camera2R']


def extract_first_number(path):
    name = os.path.basename(path)
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else -1


def make_video(image_paths, output_path, fps=25):
    if not image_paths:
        print(f"Aucune image pour {output_path}")
        return

    image_paths = sorted(image_paths, key=extract_first_number)

    first = cv2.imread(image_paths[0])
    if first is None:
        print(f"Impossible de lire la première image : {image_paths[0]}")
        return

    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Image ignorée : {path}")
            continue

        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        writer.write(img)

    writer.release()
    print(f"Vidéo sauvegardée : {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Vidéo 3D
    pose3d_images = glob.glob(os.path.join(OUTPUT_DIR, "pose3d_*.png"))
    make_video(
        pose3d_images,
        os.path.join(OUTPUT_DIR, "pose3d_video.mp4"),
        fps=FPS
    )

    # 2) Vidéos reprojetées, une par caméra
    for cam_name in CAMERA_DIRS:
        reproj_images = glob.glob(
            os.path.join(OUTPUT_DIR, f"reproj_frame*_{cam_name}.jpg")
        )
        make_video(
            reproj_images,
            os.path.join(OUTPUT_DIR, f"reprojection_{cam_name}.mp4"),
            fps=FPS
        )


if __name__ == "__main__":
    main()