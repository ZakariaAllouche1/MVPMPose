import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Import du tracker BoxMOT (DeepOCSORT)
from boxmot import DeepOCSORT

def test_boxmot_gpu():
    # --- CONFIGURATION ---
    # Remplace par le vrai chemin de ta vidéo de test
    video_path = r"\\intram\Paris\Labo\IBHGC\PROJETS\metasport\DATA\METASPORT11\Session1\Files\fichiers_video\Bruts\activity.CAM03.avi"
    
    # Ton modèle de détection YOLO de base
    model_path = "yolo26x.pt" 
    
    # --- INITIALISATION ---
    print("Chargement de YOLO sur GPU...")
    yolo_model = YOLO(model_path)
    
    print("Chargement de BoxMOT (DeepOCSORT + OSNet) sur GPU...")
    # Il va télécharger les poids OSNet automatiquement la première fois
    tracker = DeepOCSORT(
        model_weights=Path('osnet_x0_25_msmt17.pt'), # Modèle d'apparence
        device='cuda:0', # Utilisation du GPU
        fp16=True,       # Accélération matérielle activée
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible de lire la vidéo {video_path}")
        return

    # Préparation de la vidéo de sortie
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = "resultat_boxmot_gpu.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count = 0
    print("Début du tracking à pleine vitesse...")

    # --- BOUCLE PRINCIPALE ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Détection YOLO pure (sans le tracker ByteTrack intégré)
        results = yolo_model.predict(frame, classes=[0], conf=0.3, verbose=False, device='cuda:0')
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confs = results[0].boxes.conf.cpu().numpy()  # Confiance
        classes = results[0].boxes.cls.cpu().numpy() # Classe (0 = personne)
        
        # 2. Formatage des détections pour BoxMOT -> [x1, y1, x2, y2, conf, class]
        if len(boxes) > 0:
            dets = np.column_stack((boxes, confs, classes))
        else:
            dets = np.empty((0, 6))

        # 3. Mise à jour de BoxMOT
        # BoxMOT gère en interne le filtre de Kalman, OSNet et l'association
        tracks = tracker.update(dets, frame) 
        
        # 4. Dessin des résultats
        # tracks renvoie un tableau où chaque ligne est : [x1, y1, x2, y2, track_id, conf, class, idx]
        for t in tracks:
            x1, y1, x2, y2, track_id = map(int, t[:5])
            
            # Dessiner la boîte englobante
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Afficher l'ID
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        out.write(frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"Progression : {frame_count} frames traitées...")

    cap.release()
    out.release()
    print(f"Test terminé avec succès ! Regarde la vidéo : {output_path}")

if __name__ == "__main__":
    test_boxmot_gpu()