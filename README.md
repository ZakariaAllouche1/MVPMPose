# README — Guide utilisateur de `pipeline`

Ce guide explique comment installer l’environnement, configurer les chemins, lancer la pipeline complète et relancer uniquement le post-traitement si les résultats bruts existent déjà.

---

## 1. Préparer l’environnement Conda

Ouvrir un terminal dans le dossier parent du projet, puis aller dans le dossier `pipeline` :

```powershell
cd .\pipeline
```

Créer un environnement Conda dédié :

```powershell
conda create -n mvpm-pipeline python=3.10 -y
```

Activer l’environnement :

```powershell
conda activate mvpm-pipeline
```

Mettre `pip` à jour :

```powershell
python -m pip install --upgrade pip
```

Installer les dépendances du projet depuis `requirements.txt` :

```powershell
pip install -r requirements.txt
```

Si certaines dépendances GPU, CUDA, PyTorch ou ONNXRuntime nécessitent une version spécifique sur la machine, garder la même installation que celle déjà utilisée pour faire tourner la pipeline.

Pour vérifier que l’environnement est actif :

```powershell
python --version
pip list
```

---

## 2. Structure minimale attendue

Le projet doit avoir une structure de ce type :

```text
pipeline/
├── configs/
│   └── local.py
├── scripts/
│   ├── run_pipeline_full_filtered.py
│   └── filter_butterworth_and_rerender.py
├── src/
├── models/
├── requirements.txt
└── METASPORT12_undistort/
```

Les modèles doivent être placés dans le dossier configuré dans `configs/local.py`, par exemple :

```text
pipeline/models/
```

---

## 3. Configurer `configs/local.py`

Le fichier principal de configuration est :

```text
pipeline/configs/local.py
```

Les éléments les plus importants à vérifier sont :

```python
subject_name = "METASPORT12"
```

```python
base_videos_dir = Path(
    "pipeline/METASPORT12_undistort/Session1/Files/fichiers_video/Bruts"
)
```

```python
output_root = project_root / "outputs_s12_butterworth"
```

```python
camera_names = ["cam1", "cam2", "cam3", "cam4"]
```

```python
device_id_by_camera = {
    "cam1": "51433950",
    "cam2": "51434015",
    "cam3": "51470146",
    "cam4": "51470940",
}
```

Vérifier aussi les chemins des modèles :

```python
yolo_model_path = models_dir / "yolo" / "best.pt"
reid_model_path = models_dir / "reid" / "osnet_ain_x1_0_M.onnx"
alg_weights_path = multihead_weights_dir / "best_alg_multihead.pth"
vol_weights_path = multihead_weights_dir / "best_vol_multihead_sarandi_full.pth"
```

---

## 4. Choisir les activités à traiter

Dans `configs/local.py`, utiliser :

```python
activity_mode = "all"
selected_activity_name = "CMJump_03"
```

Pour traiter toutes les activités :

```python
activity_mode = "all"
```

Pour tester une seule activité :

```python
activity_mode = "single"
selected_activity_name = "CMJump_03"
```

Le mode `single` est recommandé pour tester une correction rapidement avant de relancer tout le sujet.

---

## 5. Lancer la pipeline complète

Depuis le dossier parent du projet, lancer :

```powershell
python .\pipeline\scripts\run_pipeline_full_filtered.py
```

Ce script exécute automatiquement, activité par activité :

```text
pipeline brute
→ export CSV 3D brut
→ filtre Butterworth
→ rendu vidéo filtré
→ rendu visu3D filtré
```

Dans les logs, il faut voir une alternance de ce type :

```text
===== [1/2] Pipeline brute: CMJump_01 =====
...
===== [2/2] Butterworth + rerender filtré: CMJump_01 =====
```

Puis la même chose pour les autres activités.

---

## 6. Relancer seulement le filtrage et le rendu

Si la pipeline brute a déjà été lancée et que les CSV existent déjà, il n’est pas nécessaire de relancer toute la chaîne ML.

Utiliser seulement :

```powershell
python .\pipeline\scripts\filter_butterworth_and_rerender.py
```

Cette commande relit les CSV existants et régénère les sorties filtrées.

Elle est utile après une modification liée au rendu, au GT, aux vidéos, à la visu3D ou au post-traitement Butterworth.

Elle ne relance pas :

```text
YOLO
RTMPose
tracking
ReID
learned pose
```

---

## 7. Sorties générées

Le dossier de sortie est défini dans `configs/local.py` :

```python
output_root = project_root / "outputs_s12_butterworth"
```

Les sorties principales sont :

```text
outputs_s12_butterworth/csv/
```

CSV 3D bruts produits par la pipeline.

```text
outputs_s12_butterworth/csv_butterworth/
```

CSV 3D filtrés.

```text
outputs_s12_butterworth/videos_butterworth/
```

Vidéos finales filtrées.

```text
outputs_s12_butterworth/vis3d_butterworth/
```

Visualisations 3D finales filtrées.

Les dossiers suivants peuvent aussi exister selon la configuration :

```text
outputs_s12_butterworth/videos/
outputs_s12_butterworth/vis3d/
outputs_s12_butterworth/reprojection/
outputs_s12_butterworth/debug/
```

---

## 8. Paramètres Butterworth principaux

Les paramètres sont dans `configs/local.py` :

```python
butterworth_enabled = True
butterworth_order = 4
butterworth_cutoff_mode = "activity_profile"
butterworth_default_cutoff_hz = 6.0
```

Le filtre est appliqué après la génération des CSV 3D bruts. Il ne modifie pas la pipeline ML.

Les cutoffs sont définis par famille d’activité :

```python
butterworth_activity_cutoff_hz = {
    "CMJump": 9.0,
    "SquatJump": 9.0,
    "Deadlift": 4.0,
    "Deadlift_Polystyrene": 4.0,
    "DeadliftPolystyrene": 4.0,
    "EpauleDebout": 5.0,
    "EpauleDebout_Polystyrene": 5.0,
    "EpauleDeboutPolystyrene": 5.0,
    "SledPush": 7.0,
    "SprintTapis": 10.0,
    "Sprint": 10.0,
}
```

Les petits trous sont interpolés avant filtrage :

```python
butterworth_min_segment_frames = 30
butterworth_max_gap_frames = 5
```

---

## 9. Utilisation du GT

Le GT est utilisé pendant le rerender filtré pour comparer les prédictions aux données MetaSport.

Les paramètres principaux sont :

```python
gt_show_comparison = True
gt_render_on_filtered_videos = True
gt_render_on_filtered_vis3d = True
```

Le GT venant de `points3D_t.mat` est en mètres, alors que la projection et la pipeline utilisent des millimètres. Garder donc :

```python
gt_coordinate_scale = 1000.0
gt_coordinate_offset = (0.0, 0.0, 0.0)
```

Une fois l’affichage validé, les logs de debug GT peuvent être désactivés :

```python
gt_debug_coordinates = False
```

---

## 10. Workflow recommandé

Pour tester une activité :

1. Dans `configs/local.py` :

```python
activity_mode = "single"
selected_activity_name = "CMJump_03"
```

2. Si les CSV bruts n’existent pas encore :

```powershell
python .\pipeline\scripts\run_pipeline_full_filtered.py
```

3. Si les CSV bruts existent déjà et que seul le rendu doit être refait :

```powershell
python .\pipeline\scripts\filter_butterworth_and_rerender.py
```

4. Vérifier les résultats dans :

```text
outputs_s12_butterworth/videos_butterworth/
outputs_s12_butterworth/vis3d_butterworth/
```

5. Quand tout est validé, remettre :

```python
activity_mode = "all"
```

6. Relancer sur tout le sujet :

```powershell
python .\pipeline\scripts\run_pipeline_full_filtered.py
```

ou seulement le rerender si les CSV existent déjà :

```powershell
python .\pipeline\scripts\filter_butterworth_and_rerender.py
```

---

## 11. Problèmes fréquents

### Les CSV sont manquants

Message possible :

```text
missing CSV, skipped
```

Cela signifie généralement que les CSV bruts n’ont pas encore été générés ou que `csv_dir` ne pointe pas vers le bon dossier.

Vérifier :

```python
csv_dir = output_root / "csv"
```

Puis lancer la pipeline complète :

```powershell
python .\pipeline\scripts\run_pipeline_full_filtered.py
```

---

### Le GT apparaît comme un point

Vérifier dans `configs/local.py` :

```python
gt_coordinate_scale = 1000.0
```

Si cette valeur est incorrecte, le GT peut être affiché avec une mauvaise échelle.

---

### Les bbox ou IDs de certaines personnes disparaissent dans les vidéos filtrées

Vérifier que les exports debug sont activés :

```python
save_debug_exports = True
```

Le rerender utilise ces fichiers pour récupérer les bbox et IDs des personnes trackées même lorsqu’elles ne sont pas triangulées.

---

### Une activité prend le cutoff par défaut au lieu du cutoff voulu

Vérifier que le nom de l’activité est présent dans :

```python
butterworth_activity_cutoff_hz
```

Par exemple, pour le sprint, garder aussi l’alias :

```python
"Sprint": 10.0
```

---

## 12. Commandes principales

Créer l’environnement :

```powershell
cd .\pipeline
conda create -n mvpm-pipeline python=3.10 -y
conda activate mvpm-pipeline
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Lancer la pipeline complète :

```powershell
python .\pipeline\scripts\run_pipeline_full_filtered.py
```

Relancer seulement Butterworth + rendu :

```powershell
python .\pipeline\scripts\filter_butterworth_and_rerender.py
```

Tester une seule activité dans `configs/local.py` :

```python
activity_mode = "single"
selected_activity_name = "CMJump_03"
```

Revenir à toutes les activités :

```python
activity_mode = "all"
```