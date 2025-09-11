import cv2
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
from insightface.app import FaceAnalysis
from tqdm import tqdm
import hdbscan

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except AttributeError:
    pass

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def _win_long(path: Path) -> str:
    p = str(path.resolve())
    return "\\\\?\\" + p if not p.startswith("\\\\?\\") else p

def imread_safe(path: Path):
    try:
        data = np.fromfile(_win_long(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def build_plan_live(input_dir: Path, det_size=(1024, 1024), min_cluster_size=3, min_samples=1, min_prob_threshold=0.85, progress_callback=None):
    input_dir = Path(input_dir)
    all_images = [p for p in input_dir.rglob("*") if is_image(p)]

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=det_size)

    embeddings = []
    owners = []
    unreadable = []
    no_faces = []

    for i, p in enumerate(all_images):
        img = imread_safe(p)
        if img is None:
            unreadable.append(p)
            continue
        faces = app.get(img)
        if not faces:
            no_faces.append(p)
            continue
        for f in faces:
            if getattr(f, "normed_embedding", None) is None:
                continue
            emb = np.asarray(f.normed_embedding, dtype=np.float32).reshape(1, -1)
            emb = normalize(emb, norm='l2')[0]
            embeddings.append(emb)
            owners.append(p)

        if progress_callback:
            percent = int((i + 1) / len(all_images) * 100)
            bar = int(percent / 2) * "█"
            progress_callback.text(f"📷 Scanning: {percent}%|{bar:<50}| {i+1}/{len(all_images)}")

    if not embeddings:
        return {
            "clusters": {},
            "plan": [],
            "unreadable": [str(p) for p in unreadable],
            "no_faces": [str(p) for p in no_faces],
        }

    X = np.vstack(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean", prediction_data=True)
    labels = clusterer.fit_predict(X)
    probabilities = clusterer.probabilities_

    cluster_map = {}
    cluster_by_img = {}
    for lbl, path, prob in zip(labels, owners, probabilities):
        if lbl == -1 or prob < min_prob_threshold:
            continue
        cluster_map.setdefault(int(lbl), set()).add(path)
        cluster_by_img.setdefault(path, set()).add(int(lbl))

    # фильтрация: удаляем общие фото, если кластер меньше min_cluster_size
    cluster_sizes = {k: len(v) for k, v in cluster_map.items()}
    plan = []
    for path in all_images:
        clusters = cluster_by_img.get(path, set())
        valid_clusters = [cid for cid in clusters if cluster_sizes.get(cid, 0) >= min_cluster_size]
        if valid_clusters:
            plan.append({
                "path": str(path),
                "cluster": sorted(valid_clusters),
            })

    return {
        "clusters": {int(k): [str(p) for p in sorted(v, key=lambda x: str(x))] for k, v in cluster_map.items() if cluster_sizes[k] >= min_cluster_size},
        "plan": plan,
        "unreadable": [str(p) for p in unreadable],
        "no_faces": [str(p) for p in no_faces],
    }

