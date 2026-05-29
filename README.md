# 🎯 VisionAI — Système Intelligent d'Annotation d'Images

> Projet Universitaire · Détection d'objets avec **YOLOv8** et **FastAPI**

---

## 📁 Structure du projet

```
project/
├── main.py               ← Backend FastAPI (routes, logique YOLO)
├── requirements.txt      ← Dépendances Python
├── yolov8n.pt            ← Poids du modèle (téléchargé auto au 1er lancement)
├── uploads/              ← Images originales reçues
├── results/              ← Images annotées sauvegardées
├── templates/
│   └── index.html        ← Interface HTML principale
└── static/
    └── style.css         ← Feuille de styles dashboard
```

---

## ⚙️ Installation

### 1. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

> **Note GPU** : Pour utiliser un GPU NVIDIA, remplacez `torch` dans requirements.txt par :
> `torch==2.0.0+cu118` (CUDA 11.8) et ajoutez l'index PyPI correspondant.

### 3. Télécharger le modèle YOLOv8 (automatique)

Au premier lancement, `ultralytics` télécharge automatiquement `yolov8n.pt`
(modèle nano, ~6 Mo) dans le dossier du projet.

Vous pouvez aussi le télécharger manuellement :
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## 🚀 Lancement

```bash
cd project/
python main.py
```

Ou avec `uvicorn` directement :
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Ouvrez ensuite : **http://localhost:8000**

---

## 🌐 Routes API

| Méthode | Route                  | Description                            |
|---------|------------------------|----------------------------------------|
| GET     | `/`                    | Page HTML principale                   |
| POST    | `/detect`              | Upload + détection YOLOv8 → JSON       |
| GET     | `/download/{filename}` | Téléchargement de l'image annotée      |
| GET     | `/results/{filename}`  | Accès direct aux images annotées       |
| GET     | `/uploads/{filename}`  | Accès direct aux images originales     |

### Exemple de réponse `/detect`

```json
{
  "success": true,
  "result_url": "/results/abc123_annotated.jpg",
  "original_url": "/uploads/abc123_original.jpg",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.873,
      "bbox": [125.3, 48.0, 340.7, 512.0]
    }
  ],
  "count": 1,
  "inference_time": 0.214
}
```

---

## 🎨 Fonctionnalités UI

| Fonctionnalité            | Description                                      |
|---------------------------|--------------------------------------------------|
| ⬆ Drag & Drop             | Glisser-déposer ou clic pour sélectionner        |
| 🖼 Prévisualisation        | Affichage immédiat de l'image choisie            |
| ◉ Détection               | Appel YOLOv8, barre de progression               |
| ◈ Résultat annoté         | Image avec bounding boxes colorées               |
| 📊 Métriques               | Nombre de détections, temps, confiance moyenne   |
| ⇄ Comparaison             | Toggle entre image originale et annotée          |
| ↓ Téléchargement          | Sauvegarde locale de l'image annotée             |

---

## 🧠 Technologies

- **YOLOv8n** (Ultralytics) — détection temps réel, 80 classes COCO
- **FastAPI** — serveur REST asynchrone Python
- **Jinja2** — moteur de templates HTML
- **Pillow / OpenCV** — manipulation d'images
- **Syne + DM Sans** — typographie dashboard

---

## 📝 Auteurs

Projet Universitaire — Intelligence Artificielle & Vision par Ordinateur — 2025-2026
