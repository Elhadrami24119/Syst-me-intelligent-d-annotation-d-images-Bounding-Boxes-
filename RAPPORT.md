# Rapport de Projet — Système Intelligent d'Annotation d'Images par Bounding Boxes

**Projet Universitaire · Intelligence Artificielle & Vision par Ordinateur**
**Année : 2024**

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Objectifs du projet](#2-objectifs-du-projet)
3. [Architecture du système](#3-architecture-du-système)
4. [Technologies et outils utilisés](#4-technologies-et-outils-utilisés)
5. [Les modèles d'intelligence artificielle](#5-les-modèles-dintelligence-artificielle)
6. [Structure du projet](#6-structure-du-projet)
7. [Description du backend](#7-description-du-backend)
8. [Description du frontend](#8-description-du-frontend)
9. [Modes de détection](#9-modes-de-détection)
10. [Pipeline de traitement](#10-pipeline-de-traitement)
11. [Interface utilisateur](#11-interface-utilisateur)
12. [Résultats et performances](#12-résultats-et-performances)
13. [Conclusion et perspectives](#13-conclusion-et-perspectives)

---

## 1. Introduction

Ce projet consiste en la conception et le développement d'un **système intelligent d'annotation d'images par bounding boxes**, capable de détecter et localiser automatiquement des objets dans des images numériques. Le système repose sur des modèles de deep learning basés sur l'architecture **YOLOv8** (You Only Look Once, version 8), reconnus pour leur rapidité et leur précision dans les tâches de détection d'objets en temps réel.

L'application couvre trois domaines de détection distincts :
- La **détection générale** d'objets courants (personnes, véhicules, animaux, mobilier…)
- La **détection de types d'arbres** agricoles (Acacia, Mango, Olive, Palm)
- La **détection de maladies** affectant les arbres (Anthracnose, Powdery Mildew, Fusarium Wilt, Leaf Blight, Olive Knot, Olive Leaf Spot)

---

## 2. Objectifs du projet

- Développer une application web complète de détection d'objets par bounding boxes
- Intégrer plusieurs modèles YOLOv8 spécialisés dans une seule interface
- Permettre à l'utilisateur de choisir le mode de détection adapté à son besoin
- Afficher une réponse claire : **OUI** l'objet est présent / **NON** il est absent
- Annoter visuellement les images avec des bounding boxes colorées
- Fournir des métriques de performance (temps d'inférence, confiance)

---

## 3. Architecture du système

### Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────┐
│                     NAVIGATEUR WEB                          │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Interface HTML/CSS/JS                  │   │
│   │                                                     │   │
│   │  [🔍 Général] [🌳 Arbres] [🦠 Maladies]            │   │
│   │  Drag & Drop ──► Prévisualisation                   │   │
│   │  Filtre de classe ──► Slider confiance              │   │
│   │  Bouton "Détecter" ──► Bannière OUI/NON             │   │
│   └──────────────────────┬──────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────┘
                          │ HTTP POST /detect
                          │ FormData : file + mode + target + confidence
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   SERVEUR FASTAPI (Python)                  │
│                                                             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │              Routeur de mode                         │  │
│   │                                                      │  │
│   │  mode="general"   ──►  yolov8n.pt  (COCO 80 cls)    │  │
│   │  mode="trees"     ──►  best.pt     (4 arbres)        │  │
│   │  mode="diseases"  ──►  best2.pt    (6 maladies)      │  │
│   └──────────────────────┬───────────────────────────────┘  │
│                          │                                  │
│   ┌──────────────────────▼───────────────────────────────┐  │
│   │           Pipeline de traitement                     │  │
│   │                                                      │  │
│   │  1. Validation image (extension, intégrité)          │  │
│   │  2. Sauvegarde dans /uploads/                        │  │
│   │  3. Lecture OpenCV (BGR)                             │  │
│   │  4. Inférence YOLO (modèle sélectionné)              │  │
│   │  5. Filtrage par classe cible (optionnel)            │  │
│   │  6. Dessin bounding boxes (OpenCV)                   │  │
│   │  7. Sauvegarde image annotée dans /results/          │  │
│   │  8. Réponse JSON                                     │  │
│   └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Flux de données

```
Image (JPG/PNG/WEBP/BMP)
        │
        ▼
  Validation ──► Erreur 400 si invalide
        │
        ▼
  Sauvegarde /uploads/{id}_original.ext
        │
        ▼
  cv2.imread() ──► Matrice NumPy BGR
        │
        ▼
  YOLO.predict() ──► Bounding Boxes + Classes + Scores
        │
        ▼
  Filtrage (si target fourni)
        │
        ▼
  cv2.rectangle() + cv2.putText() ──► Image annotée
        │
        ▼
  cv2.imwrite() ──► /results/{id}_annotated.jpg
        │
        ▼
  JSONResponse ──► Frontend
```

---

## 4. Technologies et outils utilisés

### 4.1 Backend

| Technologie | Logo | Version | Rôle |
|---|---|---|---|
| **Python** | 🐍 | 3.12 | Langage principal du backend |
| **FastAPI** | ⚡ | 0.111.0 | Framework web asynchrone REST |
| **Uvicorn** | 🦄 | 0.29.0 | Serveur ASGI pour FastAPI |
| **Ultralytics YOLOv8** | 🎯 | 8.2.0 | Moteur de détection d'objets |
| **OpenCV** | 👁 | ≥4.9.0 | Lecture, dessin et sauvegarde d'images |
| **PyTorch** | 🔥 | ≥2.0.0 | Framework deep learning (base de YOLO) |
| **Pillow** | 🖼 | 10.3.0 | Manipulation d'images |
| **NumPy** | 🔢 | ≥1.24.0 | Calcul matriciel |
| **Jinja2** | 📄 | 3.1.4 | Moteur de templates HTML |

### 4.2 Frontend

| Technologie | Logo | Rôle |
|---|---|---|
| **HTML5** | 🌐 | Structure de l'interface |
| **CSS3** | 🎨 | Styles, animations, responsive |
| **JavaScript (Vanilla)** | 📜 | Logique client, appels API, DOM |
| **Google Fonts** | 🔤 | Typographies Syne + DM Sans |

### 4.3 Modèles IA

| Modèle | Fichier | Domaine | Classes |
|---|---|---|---|
| YOLOv8n (COCO) | `yolov8n.pt` | Objets généraux | 80 classes |
| Custom Tree | `best.pt` | Arbres agricoles | 4 classes |
| Custom Disease | `best2.pt` | Maladies des arbres | 6 classes |

---

## 5. Les modèles d'intelligence artificielle

### 5.1 YOLOv8 — Architecture générale

**YOLO** (You Only Look Once) est une famille d'algorithmes de détection d'objets en temps réel. Contrairement aux approches en deux étapes (R-CNN), YOLO traite l'image en **un seul passage** à travers le réseau de neurones, ce qui le rend extrêmement rapide.

**YOLOv8** (2023, Ultralytics) introduit plusieurs améliorations :
- Architecture **backbone CSPDarknet** améliorée
- Tête de détection **découplée** (séparation classification/localisation)
- Perte **Distribution Focal Loss** pour une meilleure précision des boîtes
- Support natif de la segmentation, pose estimation et classification

**Principe de fonctionnement :**
1. L'image est divisée en une grille S×S
2. Chaque cellule prédit B bounding boxes avec leurs scores de confiance
3. Chaque box est définie par (x, y, w, h, confiance, classes)
4. NMS (Non-Maximum Suppression) élimine les détections redondantes

### 5.2 Modèle 1 — yolov8n.pt (Détection Générale)

- **Type** : YOLOv8 Nano (le plus léger de la famille)
- **Dataset d'entraînement** : COCO (Common Objects in Context) — 118 000 images
- **Nombre de classes** : 80
- **Classes principales** : person, car, bicycle, dog, cat, chair, bottle, cell phone, laptop, bus, truck, airplane…
- **Paramètres** : ~3.2M
- **Vitesse** : ~1ms par image sur GPU

### 5.3 Modèle 2 — best.pt (Détection d'Arbres)

- **Type** : YOLOv8 custom, entraîné par fine-tuning
- **Dataset** : Images d'arbres agricoles annotées manuellement
- **Nombre de classes** : 4
  - `Acacia-Tree` : Arbre d'acacia, reconnaissable à sa forme étalée
  - `Mango-Tree` : Manguier, feuillage dense et arrondi
  - `Olive-Tree` : Olivier, feuilles argentées caractéristiques
  - `Palm-Tree` : Palmier, silhouette verticale avec palmes
- **Application** : Agriculture de précision, inventaire forestier, cartographie

### 5.4 Modèle 3 — best2.pt (Détection de Maladies)

- **Type** : YOLOv8 custom, entraîné sur images de feuilles et branches malades
- **Nombre de classes** : 6
  - `Anthracnose` : Taches nécrotiques sombres sur feuilles et fruits
  - `Powdery-Mildew` : Oïdium — dépôt blanc poudreux sur les feuilles
  - `Fusarium-Wilt` : Flétrissement vasculaire causé par le champignon Fusarium
  - `Leaf-Blight` : Brûlure foliaire — jaunissement et nécrose des feuilles
  - `Olive-Knot` : Tumeurs bactériennes sur les branches d'olivier
  - `Olive-Leaf-Spot` : Taches circulaires sur les feuilles d'olivier
- **Application** : Diagnostic phytosanitaire, agriculture intelligente

---

## 6. Structure du projet

```
S4/
├── YOLOv8.py              ← Backend FastAPI (point d'entrée)
├── yolov8n.pt             ← Poids modèle COCO (téléchargé automatiquement)
├── best.pt                ← Poids modèle arbres (custom)
├── best2.pt               ← Poids modèle maladies (custom)
├── requirements.txt       ← Dépendances Python
├── RAPPORT.md             ← Ce rapport
│
├── templates/
│   └── index.html         ← Interface web principale (HTML + CSS + JS)
│
├── static/
│   └── style.css          ← Feuille de styles globale (dark theme)
│
├── uploads/               ← Images originales uploadées (auto-créé)
└── results/               ← Images annotées générées (auto-créé)
```

---

## 7. Description du backend

### 7.1 Initialisation

Au démarrage du serveur, les trois modèles sont chargés en mémoire :

```python
coco_model    = YOLO("yolov8n.pt")   # Modèle général COCO
tree_model    = YOLO("best.pt")      # Modèle arbres
disease_model = YOLO("best2.pt")     # Modèle maladies
```

Chaque chargement est protégé par un `try/except` : si un modèle custom est absent, le serveur démarre quand même avec les modèles disponibles.

### 7.2 Routes API

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Sert la page HTML principale |
| `POST` | `/detect` | Détection principale (image + mode + target + confidence) |
| `GET` | `/classes` | Liste toutes les classes détectables |
| `GET` | `/download/{filename}` | Téléchargement de l'image annotée |
| `GET` | `/results/{filename}` | Accès direct aux images annotées |
| `GET` | `/uploads/{filename}` | Accès direct aux images originales |

### 7.3 Route principale `/detect`

**Paramètres FormData :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `file` | UploadFile | — | Image à analyser |
| `mode` | string | `"general"` | Mode de détection |
| `target` | string | `""` | Classe à filtrer (optionnel) |
| `confidence` | float | `0.25` | Seuil de confiance [0.05–0.95] |

**Réponse JSON :**

```json
{
  "success": true,
  "detection_id": "a1b2c3d4e5",
  "result_url": "/results/a1b2c3d4e5_annotated.jpg",
  "original_url": "/uploads/a1b2c3d4e5_original.jpg",
  "detections": [
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.874,
      "bbox": [125.3, 48.0, 340.7, 512.0],
      "source": "coco"
    }
  ],
  "count": 1,
  "inference_time": 0.214,
  "target_input": "voiture",
  "found_target": true,
  "mode": "general"
}
```

### 7.4 Dessin des bounding boxes

Les bounding boxes sont dessinées avec OpenCV selon un code couleur :

| Source | Couleur | Code BGR |
|---|---|---|
| COCO (général) | Cyan / Vert / Orange… | Cyclique selon class_id |
| Arbres | Vert forêt | `(34, 197, 94)` |
| Maladies | Rouge | `(255, 80, 80)` |

Chaque box affiche : `NomClasse Confiance%`

---

## 8. Description du frontend

### 8.1 Structure HTML

L'interface est organisée en **deux colonnes** :
- **Colonne gauche** : Sélection du mode, upload d'image, filtres, bouton détecter, bannière résultat
- **Colonne droite** : Image annotée, métriques, liste des détections, boutons d'action

### 8.2 Design

- **Thème** : Dark (fond `#0d0f14`)
- **Palette** :
  - Cyan `#00e5ff` — accent principal, mode général
  - Vert `#22c55e` — mode arbres
  - Rouge `#f87171` — mode maladies
  - Vert détection `#39ff85` — résultat positif
  - Rouge résultat `#ff4d6d` — résultat négatif
- **Typographie** : Syne (titres, boutons) + DM Sans (corps de texte)
- **Responsive** : 2 colonnes ≥768px, 1 colonne <768px

### 8.3 Composants JavaScript

| Fonction | Rôle |
|---|---|
| `setMode(mode)` | Change le mode actif, met à jour les boutons et chips |
| `loadFile(file)` | Charge et prévisualise l'image sélectionnée |
| `detectObjects()` | Envoie la requête POST /detect avec FormData |
| `showAnswer(data)` | Affiche la bannière OUI/NON selon `found_target` |
| `displayResults(data)` | Affiche l'image annotée, métriques et liste |

---

## 9. Modes de détection

### Mode 🔍 Général — `yolov8n.pt`

Utilise le modèle YOLOv8 pré-entraîné sur le dataset COCO. Détecte 80 catégories d'objets du quotidien. Les bounding boxes sont dessinées en couleurs cycliques (cyan, vert, orange, violet, jaune, rose).

**Cas d'usage** : Surveillance, comptage de personnes, détection de véhicules, analyse de scènes.

### Mode 🌳 Arbres — `best.pt`

Utilise un modèle custom entraîné spécifiquement sur des images d'arbres agricoles. Toutes les bounding boxes sont en **vert forêt**. L'utilisateur peut filtrer par espèce (Acacia, Mango, Olive, Palm).

**Cas d'usage** : Inventaire forestier, agriculture de précision, cartographie végétale.

### Mode 🦠 Maladies — `best2.pt`

Utilise un modèle custom entraîné sur des images de feuilles et branches malades. Toutes les bounding boxes sont en **rouge**. L'utilisateur peut filtrer par type de maladie.

**Cas d'usage** : Diagnostic phytosanitaire, détection précoce de maladies, aide à la décision agricole.

---

## 10. Pipeline de traitement

```
Étape 1 : Réception de la requête HTTP POST
          ├── Validation de l'extension (.jpg, .jpeg, .png, .webp, .bmp)
          └── Validation du mode ("general", "trees", "diseases")

Étape 2 : Génération d'un identifiant unique (UUID 10 caractères)
          └── Nommage : {id}_original.ext / {id}_annotated.jpg

Étape 3 : Sauvegarde de l'image originale dans /uploads/

Étape 4 : Lecture avec OpenCV
          └── cv2.imread() → matrice NumPy BGR (H × W × 3)

Étape 5 : Inférence YOLO
          ├── Sélection du modèle selon le mode
          ├── model.predict(conf=..., iou=0.45)
          └── Extraction : class_id, class_name, confidence, bbox (xyxy)

Étape 6 : Filtrage optionnel
          └── Si target fourni → ne garder que les détections correspondantes

Étape 7 : Annotation visuelle
          ├── cv2.rectangle() → bounding box colorée (épaisseur 2px)
          ├── cv2.getTextSize() → calcul de la taille du label
          ├── cv2.rectangle() → fond coloré du label
          └── cv2.putText() → texte "NomClasse Conf%"

Étape 8 : Sauvegarde de l'image annotée dans /results/

Étape 9 : Construction et envoi de la réponse JSON
```

---

## 11. Interface utilisateur

### Sélecteur de mode

Trois boutons visuels permettent de choisir le modèle avant l'analyse :

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│      🔍      │  │      🌳      │  │      🦠      │
│   GÉNÉRAL    │  │    ARBRES    │  │   MALADIES   │
│ 80 cls COCO  │  │ Acacia·Mango │  │  6 maladies  │
│ yolov8n.pt   │  │ Olive·Palm   │  │  agricoles   │
└──────────────┘  └──────────────┘  └──────────────┘
  [Actif=Cyan]     [Actif=Vert]      [Actif=Rouge]
```

### Bannière de résultat

Après chaque détection avec un filtre de classe :

```
✅  OUI — « voiture » est dans l'image
    3 détection(s) · confiance 87%

❌  NON — « voiture » n'est pas dans l'image
    Essayez un seuil plus bas ou vérifiez l'orthographe
```

### Métriques affichées

| Métrique | Description |
|---|---|
| Détections | Nombre total d'objets détectés |
| Temps (s) | Durée totale de l'inférence |
| Conf. moy. | Confiance moyenne de toutes les détections |

---

## 12. Résultats et performances

### Temps d'inférence typiques (CPU)

| Modèle | Temps moyen | Image 640×480 |
|---|---|---|
| yolov8n.pt (COCO) | ~0.15–0.30s | Rapide |
| best.pt (Arbres) | ~0.15–0.30s | Rapide |
| best2.pt (Maladies) | ~0.15–0.30s | Rapide |

### Formats d'images supportés

JPG · JPEG · PNG · WEBP · BMP

### Seuil de confiance

- **Défaut** : 25% — bon équilibre précision/rappel
- **Minimum** : 5% — détecte plus d'objets, plus de faux positifs
- **Maximum** : 90% — très sélectif, moins de détections

---

## 13. Conclusion et perspectives

### Réalisations

Ce projet a permis de développer une application web complète et fonctionnelle de détection d'objets par bounding boxes, intégrant trois modèles YOLOv8 spécialisés dans une interface unifiée et intuitive. Le système répond aux exigences d'un projet universitaire en intelligence artificielle et vision par ordinateur.

### Points forts

- Architecture modulaire : chaque modèle est indépendant et isolé
- Interface claire avec réponse binaire OUI/NON immédiatement visible
- Support multilingue FR/EN pour les classes COCO
- Code propre, commenté et maintenable
- Gestion robuste des erreurs (modèles absents, images invalides)

### Perspectives d'amélioration

1. **Entraînement amélioré** : Augmenter le dataset des modèles custom (best.pt, best2.pt) pour améliorer la précision
2. **GPU** : Déployer sur un serveur avec GPU NVIDIA pour des inférences en temps réel (<50ms)
3. **Vidéo** : Étendre la détection aux flux vidéo (webcam, fichiers MP4)
4. **API REST** : Documenter et exposer l'API pour intégration dans d'autres systèmes
5. **Segmentation** : Passer de la détection (bounding boxes) à la segmentation d'instance (masques précis)
6. **Mobile** : Adapter l'interface pour une utilisation sur smartphone (terrain agricole)

---

*Rapport généré automatiquement — Projet VisionAI · Système Intelligent d'Annotation d'Images*
