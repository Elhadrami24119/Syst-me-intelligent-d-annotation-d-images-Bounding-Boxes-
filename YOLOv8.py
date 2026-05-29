# ============================================================
# YOLOv8.py — Backend FastAPI
# Système Intelligent d'Annotation d'Images (Bounding Boxes)
# ============================================================
# Trois modèles :
#   1. yolov8n.pt → 80 classes COCO  (mode "general")
#   2. best.pt    → 4 types d'arbres  (mode "trees")
#   3. best2.pt   → 6 maladies        (mode "diseases")
#
# Le mode est envoyé explicitement par le frontend via FormData.
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
import shutil
import cv2

# ── Initialisation FastAPI ────────────────────────────────────
app = FastAPI(
    title="Système Intelligent d'Annotation d'Images",
    description="Détection YOLO — Général / Arbres / Maladies",
    version="4.0.0",
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

# ── Chargement modèle COCO — mode "general" ──────────────────
print("[YOLO-COCO] Chargement yolov8n.pt…")
coco_model   = YOLO(str(BASE_DIR / "yolov8n.pt"))
COCO_CLASSES = list(coco_model.names.values())
print(f"[YOLO-COCO] ✓ — {len(COCO_CLASSES)} classes")

# ── Chargement modèle arbres — mode "trees" ──────────────────
TREE_MODEL_AVAILABLE = False
tree_model           = None
try:
    tree_model = YOLO(str(BASE_DIR / "best.pt"))
    TREE_MODEL_AVAILABLE = True
    TREE_CLASSES = list(tree_model.names.values())
    print(f"[YOLO-TREE] ✓ — classes : {TREE_CLASSES}")
except Exception as e:
    TREE_CLASSES = ["Acacia-Tree", "Mango-Tree", "Olive-Tree", "Palm-Tree"]
    print(f"[YOLO-TREE] Non disponible ({e})")

# ── Chargement modèle maladies — mode "diseases" ─────────────
DISEASE_MODEL_AVAILABLE = False
disease_model           = None
try:
    disease_model = YOLO(str(BASE_DIR / "best2.pt"))
    DISEASE_MODEL_AVAILABLE = True
    DISEASE_CLASSES = list(disease_model.names.values())
    print(f"[YOLO-DISEASE] ✓ — classes : {DISEASE_CLASSES}")
except Exception as e:
    DISEASE_CLASSES = [
        "Anthracnose", "Powdery-Mildew", "Fusarium-Wilt",
        "Leaf-Blight", "Olive-Knot", "Olive-Leaf-Spot",
    ]
    print(f"[YOLO-DISEASE] Non disponible ({e})")

# ── Traduction FR → EN (mode general) ────────────────────────
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

# Couleurs bounding boxes
COCO_COLORS = [
    (0, 229, 255), (57, 255, 133), (255, 100, 50),
    (180, 50, 255), (255, 220, 0), (255, 60, 120),
]
TREE_COLOR    = (34, 197, 94)   # vert forêt
DISEASE_COLOR = (255, 80, 80)   # rouge maladie


# ── Utilitaire : résoudre la classe cible (mode general) ─────
def resolve_target(target_clean: str) -> tuple[str, bool, str]:
    """Retourne (target_en, is_coco_class, fallback_class)."""
    if not target_clean:
        return "", False, ""
    if target_clean in FR_TO_EN:
        return FR_TO_EN[target_clean], True, ""
    coco_lower = [c.lower() for c in COCO_CLASSES]
    if target_clean in coco_lower:
        return target_clean, True, ""
    matches = [c for c in COCO_CLASSES
               if re.search(r'\b' + re.escape(target_clean) + r'\b', c.lower())]
    if matches:
        return matches[0], True, ""
    fallback = ""
    for word in reversed(target_clean.split()):
        if word in FR_TO_EN:
            fallback = FR_TO_EN[word]; break
        if word in coco_lower:
            fallback = word; break
        wm = [c for c in COCO_CLASSES
              if re.search(r'\b' + re.escape(word) + r'\b', c.lower())]
        if wm:
            fallback = wm[0]; break
    return target_clean, False, fallback


# ── Routes ───────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/classes")
async def get_classes():
    return JSONResponse({
        "classes_coco"    : COCO_CLASSES,
        "classes_trees"   : TREE_CLASSES,
        "classes_diseases": DISEASE_CLASSES,
        "classes_fr"      : list(FR_TO_EN.keys()),
    })


@app.post("/detect")
async def detect_objects(
    file:       UploadFile = File(...),
    mode:       str        = Form(default="general"),   # "general" | "trees" | "diseases"
    target:     str        = Form(default=""),
    confidence: float      = Form(default=0.25),
):
    """
    Détecte les objets dans une image selon le mode choisi.

    Paramètres :
      file       : image (jpg, jpeg, png, webp, bmp)
      mode       : "general" | "trees" | "diseases"
      target     : filtre optionnel sur une classe précise
      confidence : seuil de confiance [0.05 – 0.95]

    Routage :
      mode=general   → yolov8n.pt uniquement (80 classes COCO)
      mode=trees     → best.pt uniquement    (4 types d'arbres)
      mode=diseases  → best2.pt uniquement   (6 maladies)
    """

    # 1. Validation extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Format non supporté : {ext}")

    # 2. Validation mode
    if mode not in ("general", "trees", "diseases"):
        raise HTTPException(status_code=400,
                            detail=f"Mode invalide : '{mode}'. Valeurs : general, trees, diseases")

    # 3. IDs uniques
    unique_id   = uuid.uuid4().hex[:10]
    upload_name = f"{unique_id}_original{ext}"
    result_name = f"{unique_id}_annotated.jpg"
    upload_path = UPLOAD_DIR / upload_name
    result_path = RESULT_DIR / result_name

    # 4. Sauvegarde image
    with open(upload_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # 5. Lecture OpenCV
    img_cv = cv2.imread(str(upload_path))
    if img_cv is None:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Fichier image invalide ou corrompu.")

    target_clean = target.strip().lower()
    conf         = max(0.05, min(confidence, 0.95))
    t0           = time.perf_counter()

    print(f"[DETECT] mode='{mode}' | target='{target}' | conf={conf}")

    detections = []

    # ── Mode "general" : YOLO COCO uniquement ────────────────
    if mode == "general":
        res = coco_model.predict(source=str(upload_path),
                                 conf=conf, iou=0.45,
                                 save=False, verbose=False)
        target_en, is_coco, fallback = resolve_target(target_clean)

        for box in res[0].boxes:
            cid   = int(box.cls[0])
            cname = coco_model.names[cid]
            cval  = float(box.conf[0])
            xyxy  = box.xyxy[0].tolist()

            # Filtre si target fourni
            if target_clean:
                if is_coco and cname.lower() != target_en.lower():
                    continue
                if not is_coco and fallback and cname.lower() != fallback.lower():
                    continue

            detections.append({
                "class_id"  : cid,
                "class_name": cname,
                "confidence": round(cval, 3),
                "bbox"      : [round(v, 1) for v in xyxy],
                "source"    : "coco",
            })

    # ── Mode "trees" : best.pt uniquement ────────────────────
    elif mode == "trees":
        if not TREE_MODEL_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="Modèle arbres (best.pt) non disponible.")
        res = tree_model.predict(source=str(upload_path),
                                 conf=conf, iou=0.45,
                                 save=False, verbose=False)
        for box in res[0].boxes:
            cid   = int(box.cls[0])
            cname = tree_model.names[cid]
            cval  = float(box.conf[0])
            xyxy  = box.xyxy[0].tolist()

            # Filtre si target fourni (ex: "palm", "acacia")
            if target_clean and target_clean not in cname.lower():
                continue

            detections.append({
                "class_id"  : cid,
                "class_name": cname,
                "confidence": round(cval, 3),
                "bbox"      : [round(v, 1) for v in xyxy],
                "source"    : "tree",
            })

    # ── Mode "diseases" : best2.pt uniquement ────────────────
    elif mode == "diseases":
        if not DISEASE_MODEL_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="Modèle maladies (best2.pt) non disponible.")
        res = disease_model.predict(source=str(upload_path),
                                    conf=conf, iou=0.45,
                                    save=False, verbose=False)
        for box in res[0].boxes:
            cid   = int(box.cls[0])
            cname = disease_model.names[cid]
            cval  = float(box.conf[0])
            xyxy  = box.xyxy[0].tolist()

            # Filtre si target fourni (ex: "anthracnose", "blight")
            if target_clean and target_clean not in cname.lower():
                continue

            detections.append({
                "class_id"  : cid,
                "class_name": cname,
                "confidence": round(cval, 3),
                "bbox"      : [round(v, 1) for v in xyxy],
                "source"    : "disease",
            })

    t_total      = round(time.perf_counter() - t0, 3)
    found_target = len(detections) > 0 if target_clean else None

    # 6. Dessin bounding boxes
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        if det["source"] == "tree":
            color = TREE_COLOR
        elif det["source"] == "disease":
            color = DISEASE_COLOR
        else:
            color = COCO_COLORS[det["class_id"] % len(COCO_COLORS)]

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']*100:.0f}%"
        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img_cv, (x1, y1 - lh - bl - 4), (x1 + lw + 4, y1), color, -1)
        cv2.putText(img_cv, label, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(result_path), img_cv)

    # 7. Message résultat
    if target_clean and found_target:
        search_info = f"« {target} » trouvé : {len(detections)} détection(s)."
    elif target_clean and not found_target:
        search_info = f"« {target} » non trouvé dans l'image."
    else:
        search_info = f"{len(detections)} objet(s) détecté(s)."

    return JSONResponse({
        "success"                : True,
        "detection_id"           : unique_id,
        "result_url"             : f"/results/{result_name}",
        "original_url"           : f"/uploads/{upload_name}",
        "detections"             : detections,
        "count"                  : len(detections),
        "inference_time"         : t_total,
        "target"                 : target_clean,
        "target_input"           : target,
        "search_info"            : search_info,
        "found_target"           : found_target,
        "mode"                   : mode,
        "tree_model_available"   : TREE_MODEL_AVAILABLE,
        "disease_model_available": DISEASE_MODEL_AVAILABLE,
        "tree_classes"           : TREE_CLASSES,
        "disease_classes"        : DISEASE_CLASSES,
    })


@app.get("/download/{filename}")
async def download_result(filename: str):
    file_path = RESULT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(str(file_path), media_type="image/jpeg",
                        filename=f"annotated_{filename}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("YOLOv8:app", host="0.0.0.0", port=8000, reload=True)
