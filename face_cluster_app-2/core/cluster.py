
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def imread_safe(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def build_plan(input_dir: Path, det_size=(1024, 1024), dbscan_eps=0.5, dbscan_min_samples=2):
    input_dir = Path(input_dir)
    all_images = [p for p in input_dir.rglob("*") if is_image(p)]

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=det_size)

    embeddings = []
    owners = []
    unreadable = []

    for p in tqdm(all_images, desc="📷 Scanning"):
        img = imread_safe(p)
        if img is None:
            unreadable.append(p)
            continue
        faces = app.get(img)
        for f in faces:
            if f.normed_embedding is None:
                continue
            embeddings.append(f.normed_embedding.astype(np.float32))
            owners.append(p)

    if not embeddings:
        return {
            "clusters": {},
            "plan": [],
            "unreadable": [str(p) for p in unreadable],
        }

    X = np.vstack(embeddings)
    model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine")
    labels = model.fit_predict(X)

    cluster_map = {}
    for lbl, path in zip(labels, owners):
        if lbl == -1:
            continue
        cluster_map.setdefault(int(lbl), set()).add(path)

    cluster_by_img = {}
    for lbl, path in zip(labels, owners):
        if lbl == -1:
            continue
        cluster_by_img.setdefault(path, set()).add(int(lbl))

    plan = []
    for path in all_images:
        clusters = cluster_by_img.get(path)
        if not clusters:
            continue
        plan.append({
            "path": str(path),
            "cluster": sorted(list(clusters)),
        })

    return {
        "clusters": {int(k): [str(p) for p in sorted(v)] for k, v in cluster_map.items()},
        "plan": plan,
        "unreadable": [str(p) for p in unreadable],
    }
