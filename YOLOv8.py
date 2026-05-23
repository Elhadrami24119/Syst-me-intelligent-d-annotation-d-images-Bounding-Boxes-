# ============================================================
# YOLOv8.py — Backend FastAPI
# Système Intelligent d'Annotation d'Images (Bounding Boxes)
# ============================================================
# Pipeline :
#   1. YOLO COCO  → détecte 80 classes d'objets courants
#   2. Roboflow   → détecte les arbres (modèle custom find-tree/1)
#
# Routage :
#   target vide         → YOLO + Roboflow (tous les objets)
#   target = arbre/tree → Roboflow uniquement
#   target = classe COCO → YOLO filtré sur la classe
# ============================================================

import uuid
import time
import re
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from ultralytics import YOLO
from PIL import Image
import shutil
import cv2
import numpy as np

# ── Initialisation FastAPI ────────────────────────────────────
app = FastAPI(
    title="Système Intelligent d'Annotation d'Images",
    description="Détection d'objets par Bounding Boxes — YOLOv8 + Roboflow",
    version="1.0.0",
)

# ── Chemins ───────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
UPLOAD_DIR   = BASE_DIR / "uploads"
RESULT_DIR   = BASE_DIR / "results"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR   = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# ── Fichiers statiques ────────────────────────────────────────
app.mount("/static",  StaticFiles(directory=STATIC_DIR),  name="static")
app.mount("/results", StaticFiles(directory=RESULT_DIR),  name="results")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR),  name="uploads")

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# ── Chargement YOLOv8n ───────────────────────────────────────
MODEL_PATH = BASE_DIR / "yolov8n.pt"
print(f"[YOLO] Chargement depuis : {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print("[YOLO] Modèle chargé ✓")

COCO_CLASSES = list(model.names.values())

# ── Chargement modèle Roboflow (find-tree/1) ─────────────────
ROBOFLOW_API_KEY   = "JVNPmcYSsGtctdThmLks"
ROBOFLOW_MODEL     = "find-tree/1"
ROBOFLOW_AVAILABLE = False
rf_model           = None

try:
    from inference_sdk import InferenceHTTPClient
    rf_model = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=ROBOFLOW_API_KEY,
    )
    ROBOFLOW_AVAILABLE = True
    print(f"[ROBOFLOW] Modèle '{ROBOFLOW_MODEL}' prêt ✓")
except Exception as e:
    print(f"[ROBOFLOW] Non disponible ({e})")

# Mots-clés qui déclenchent le modèle Roboflow
TREE_KEYWORDS = {
    "arbre", "arbres", "tree", "trees", "palm", "palmier",
    "oak", "chêne", "pine", "pin", "sapin", "forêt", "forest",
    "vegetation", "végétation",
}

# ── Traduction FR → EN ────────────────────────────────────────
FR_TO_EN = {
    "personne": "person", "homme": "person", "femme": "person", "gens": "person",
    "vélo": "bicycle", "velo": "bicycle", "bicyclette": "bicycle",
    "voiture": "car", "auto": "car", "automobile": "car",
    "moto": "motorcycle", "motocyclette": "motorcycle",
    "avion": "airplane", "bus": "bus", "autobus": "bus",
    "train": "train", "camion": "truck", "bateau": "boat",
    "chien": "dog", "chat": "cat", "cheval": "horse",
    "vache": "cow", "mouton": "sheep",
    "éléphant": "elephant", "elephant": "elephant",
    "ours": "bear", "zèbre": "zebra", "zebre": "zebra",
    "girafe": "giraffe", "sac à dos": "backpack", "sac": "backpack",
    "parapluie": "umbrella", "sac à main": "handbag", "cravate": "tie",
    "valise": "suitcase", "frisbee": "frisbee",
    "skis": "skis", "ski": "skis", "snowboard": "snowboard",
    "ballon": "sports ball", "balle": "sports ball",
    "cerf-volant": "kite", "batte": "baseball bat",
    "gant": "baseball glove", "skateboard": "skateboard",
    "planche de surf": "surfboard", "raquette": "tennis racket",
    "bouteille": "bottle", "verre": "wine glass", "tasse": "cup",
    "fourchette": "fork", "couteau": "knife", "cuillère": "spoon",
    "bol": "bowl", "banane": "banana", "pomme": "apple",
    "sandwich": "sandwich", "orange": "orange", "brocoli": "broccoli",
    "carotte": "carrot", "hotdog": "hot dog", "pizza": "pizza",
    "beignet": "donut", "gâteau": "cake", "chaise": "chair",
    "canapé": "couch", "sofa": "couch", "plante": "potted plant",
    "lit": "bed", "table": "dining table", "toilettes": "toilet",
    "télévision": "tv", "tv": "tv", "ecran": "tv",
    "ordinateur portable": "laptop", "souris": "mouse",
    "télécommande": "remote", "clavier": "keyboard",
    "téléphone": "cell phone", "telephone": "cell phone",
    "four": "oven", "grille-pain": "toaster", "évier": "sink",
    "réfrigérateur": "refrigerator", "frigo": "refrigerator",
    "livre": "book", "horloge": "clock", "vase": "vase",
    "ciseaux": "scissors", "ours en peluche": "teddy bear",
    "sèche-cheveux": "hair drier", "brosse à dents": "toothbrush",
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

BOX_COLORS = [
    (0, 229, 255),   # cyan
    (57, 255, 133),  # vert
    (255, 100, 50),  # orange
    (180, 50, 255),  # violet
    (255, 220, 0),   # jaune
    (255, 60, 120),  # rose
]


# ── Utilitaire : résoudre la classe cible ─────────────────────
def resolve_target(target_clean: str) -> tuple[str, bool, str]:
    """
    Retourne (target_en, is_coco_class, fallback_class).
    - target_en      : classe COCO correspondante (ou requête brute)
    - is_coco_class  : True si correspondance exacte avec une classe COCO
    - fallback_class : mot-clé COCO extrait d'une description multi-mots
    """
    if not target_clean:
        return "", False, ""

    if target_clean in FR_TO_EN:
        return FR_TO_EN[target_clean], True, ""

    coco_lower = [c.lower() for c in COCO_CLASSES]
    if target_clean in coco_lower:
        return target_clean, True, ""

    matches_word = [c for c in COCO_CLASSES
                    if re.search(r'\b' + re.escape(target_clean) + r'\b', c.lower())]
    if matches_word:
        return matches_word[0], True, ""

    # Description multi-mots → extraire le mot-clé COCO
    fallback_class = ""
    for word in reversed(target_clean.split()):
        if word in FR_TO_EN:
            fallback_class = FR_TO_EN[word]
            break
        if word in coco_lower:
            fallback_class = word
            break
        wm = [c for c in COCO_CLASSES
              if re.search(r'\b' + re.escape(word) + r'\b', c.lower())]
        if wm:
            fallback_class = wm[0]
            break

    return target_clean, False, fallback_class


# ── Utilitaire : détecter si la cible est un arbre ───────────
def is_tree_target(target_clean: str) -> bool:
    if not target_clean:
        return False
    if target_clean in TREE_KEYWORDS:
        return True
    for kw in TREE_KEYWORDS:
        if kw in target_clean:
            return True
    return False


# ── Utilitaire : inférence Roboflow ──────────────────────────
def run_roboflow(image_path: str, img_w: int, img_h: int) -> list[dict]:
    """
    Appelle le modèle Roboflow find-tree/1.
    Retourne les détections au même format que YOLO.
    """
    if not ROBOFLOW_AVAILABLE:
        return []
    try:
        result = rf_model.infer(image_path, model_id=ROBOFLOW_MODEL)

        # Log de la réponse brute pour diagnostic
        print(f"[ROBOFLOW] Réponse brute type={type(result)}")
        print(f"[ROBOFLOW] Réponse brute = {result}")

        # Extraire les prédictions selon le format retourné
        if isinstance(result, dict):
            predictions = result.get("predictions", [])
        elif isinstance(result, list):
            predictions = getattr(result[0], "predictions", []) if result else []
        else:
            predictions = getattr(result, "predictions", [])

        print(f"[ROBOFLOW] {len(predictions)} prédiction(s) brutes")

        detections = []
        for pred in predictions:
            if hasattr(pred, "x"):
                cx, cy, w, h = pred.x, pred.y, pred.width, pred.height
                conf, label  = pred.confidence, pred.class_name
            else:
                cx   = pred.get("x", 0)
                cy   = pred.get("y", 0)
                w    = pred.get("width", 0)
                h    = pred.get("height", 0)
                conf = pred.get("confidence", 0)
                label = pred.get("class", pred.get("class_name", "tree"))

            x1 = max(0, int(cx - w / 2))
            y1 = max(0, int(cy - h / 2))
            x2 = min(img_w, int(cx + w / 2))
            y2 = min(img_h, int(cy + h / 2))

            print(f"[ROBOFLOW] Détection : {label} conf={conf:.2f} bbox=({x1},{y1},{x2},{y2})")

            detections.append({
                "class_id"   : -1,
                "class_name" : label,
                "confidence" : round(float(conf), 3),
                "bbox"       : [float(x1), float(y1), float(x2), float(y2)],
                "source"     : "roboflow",
            })

        print(f"[ROBOFLOW] {len(detections)} détection(s) finales")
        return detections

    except Exception as e:
        print(f"[ROBOFLOW] Erreur : {type(e).__name__} — {e}")
        import traceback
        traceback.print_exc()
        return []


# ── Route GET / ───────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── Route GET /classes ────────────────────────────────────────
@app.get("/classes")
async def get_classes():
    """Liste des 80 classes COCO détectables + traductions FR."""
    return JSONResponse({
        "classes_en": COCO_CLASSES,
        "classes_fr": list(FR_TO_EN.keys()),
    })


# ── Route POST /detect ────────────────────────────────────────
@app.post("/detect")
async def detect_objects(
    file:       UploadFile = File(...),
    target:     str        = Form(default=""),
    confidence: float      = Form(default=0.25),
):
    """
    Détecte les objets dans une image.

    Paramètres :
      file       : image (jpg, jpeg, png, webp, bmp)
      target     : objet à rechercher — vide = tous les objets
      confidence : seuil de confiance YOLO [0.05 – 0.95]

    Routage :
      vide         → YOLO COCO + Roboflow
      arbre/tree   → Roboflow uniquement
      classe COCO  → YOLO filtré
    """

    # 1. Validation extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Format non supporté : {ext}")

    # 2. IDs uniques
    unique_id   = uuid.uuid4().hex[:10]
    upload_name = f"{unique_id}_original{ext}"
    result_name = f"{unique_id}_annotated.jpg"
    upload_path = UPLOAD_DIR / upload_name
    result_path = RESULT_DIR / result_name

    # 3. Sauvegarde image originale
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 4. Lecture unique OpenCV + conversion PIL
    #    OpenCV (BGR) → img_cv pour le dessin des boxes
    #    Conversion RGB → PIL pour compatibilité Roboflow
    img_cv = cv2.imread(str(upload_path))
    if img_cv is None:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Fichier image invalide ou corrompu.")

    img_h, img_w = img_cv.shape[:2]

    # 5. Résolution de la cible + routage
    target_clean   = target.strip().lower()
    target_en, is_coco, fallback_class = resolve_target(target_clean)
    target_is_tree = is_tree_target(target_clean)

    # Règles de routage
    run_yolo_model     = not target_is_tree
    run_roboflow_model = (not target_clean) or target_is_tree

    print(f"[DETECT] target='{target}' | is_tree={target_is_tree} | "
          f"yolo={run_yolo_model} | roboflow={run_roboflow_model}")

    # 6. Inférence YOLO
    t_start    = time.perf_counter()
    yolo_boxes = []

    if run_yolo_model:
        results    = model.predict(
            source=str(upload_path),
            conf=max(0.05, min(confidence, 0.95)),
            iou=0.45,
            save=False,
            verbose=False,
        )
        yolo_boxes = results[0].boxes

    t_yolo = round(time.perf_counter() - t_start, 3)

    # 7. Inférence Roboflow
    roboflow_detections = []
    t_roboflow = 0.0

    if run_roboflow_model and ROBOFLOW_AVAILABLE:
        t_rf = time.perf_counter()
        roboflow_detections = run_roboflow(str(upload_path), img_w, img_h)
        t_roboflow = round(time.perf_counter() - t_rf, 3)

    # 8. Filtrage des détections YOLO
    yolo_detections = []
    found_target    = False

    for box in yolo_boxes:
        class_id       = int(box.cls[0])
        class_name     = model.names[class_id]
        confidence_val = float(box.conf[0])
        xyxy           = box.xyxy[0].tolist()

        # Mode : tous les objets
        if not target_clean:
            yolo_detections.append({
                "class_id"   : class_id,
                "class_name" : class_name,
                "confidence" : round(confidence_val, 3),
                "bbox"       : [round(v, 1) for v in xyxy],
                "source"     : "yolo",
            })
            continue

        # Mode : classe COCO exacte
        if is_coco:
            if class_name.lower() != target_en.lower():
                continue
            found_target = True
            yolo_detections.append({
                "class_id"   : class_id,
                "class_name" : class_name,
                "confidence" : round(confidence_val, 3),
                "bbox"       : [round(v, 1) for v in xyxy],
                "source"     : "yolo",
            })
            continue

        # Mode : fallback sur mot-clé extrait
        if fallback_class:
            if class_name.lower() != fallback_class.lower():
                continue
            found_target = True
            yolo_detections.append({
                "class_id"   : class_id,
                "class_name" : class_name,
                "confidence" : round(confidence_val, 3),
                "bbox"       : [round(v, 1) for v in xyxy],
                "source"     : "yolo",
            })

    # Fusion des détections
    detections = yolo_detections + roboflow_detections
    t_total    = round(time.perf_counter() - t_start, 3)

    # Mise à jour found_target pour Roboflow
    if target_is_tree and roboflow_detections:
        found_target = True
    elif not target_clean:
        found_target = None  # mode "tous" → pas de notion trouvé/non trouvé

    # 9. Dessin des bounding boxes
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cid   = det["class_id"]
        color = BOX_COLORS[cid % len(BOX_COLORS)] if cid >= 0 else (255, 165, 0)

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)

        label = f"{det['class_name']} {det['confidence']*100:.0f}%"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img_cv, (x1, y1 - lh - baseline - 4), (x1 + lw + 4, y1), color, -1)
        cv2.putText(img_cv, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(result_path), img_cv)

    # 10. Message résultat
    if not target_clean:
        n_yolo = len(yolo_detections)
        n_rf   = len(roboflow_detections)
        search_info = f"{n_yolo} objet(s) COCO + {n_rf} arbre(s) détecté(s)."
        mode_used   = "yolo+roboflow" if ROBOFLOW_AVAILABLE else "yolo"
    elif target_is_tree:
        n_rf = len(roboflow_detections)
        search_info = f"Roboflow : {n_rf} arbre(s) trouvé(s)." if n_rf else "Aucun arbre détecté."
        mode_used   = "roboflow"
    elif is_coco or fallback_class:
        search_info = (f"« {target} » trouvé : {len(detections)} fois."
                       if found_target else f"« {target} » non trouvé.")
        mode_used   = "yolo"
    else:
        search_info = f"{len(detections)} objet(s) détecté(s)."
        mode_used   = "yolo"

    # 11. Réponse JSON
    return JSONResponse({
        "success"            : True,
        "detection_id"       : unique_id,
        "result_url"         : f"/results/{result_name}",
        "original_url"       : f"/uploads/{upload_name}",
        "detections"         : detections,
        "count"              : len(detections),
        "inference_time"     : t_total,
        "time_yolo"          : t_yolo,
        "time_roboflow"      : t_roboflow,
        "target"             : target_en,
        "target_input"       : target,
        "search_info"        : search_info,
        "found_target"       : found_target,
        "mode"               : mode_used,
        "roboflow_available" : ROBOFLOW_AVAILABLE,
        "roboflow_count"     : len(roboflow_detections),
    })


# ── Route GET /download/{filename} ───────────────────────────
@app.get("/download/{filename}")
async def download_result(filename: str):
    file_path = RESULT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(
        path=str(file_path),
        media_type="image/jpeg",
        filename=f"annotated_{filename}",
    )


# ── Point d'entrée ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("YOLOv8:app", host="0.0.0.0", port=8000, reload=True)
