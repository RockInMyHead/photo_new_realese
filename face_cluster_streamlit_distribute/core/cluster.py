import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from insightface.app import FaceAnalysis
from hdbscan import HDBSCAN
from io import StringIO
import sys

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

def build_plan(input_dir: Path, det_size=(1024, 1024), min_cluster_size=3):
    input_dir = Path(input_dir)
    all_images = [p for p in input_dir.rglob("*") if is_image(p)]

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=det_size)

    embeddings = []
    owners = []
    unreadable = []
    no_faces = []

    tqdm_output = StringIO()
    old_stdout = sys.stdout
    sys.stdout = tqdm_output

    try:
        for p in tqdm(all_images, desc="📷 Scanning"):
            img = imread_safe(p)
            if img is None:
                unreadable.append(p)
                continue

            faces = app.get(img)
            if not faces:
                no_faces.append(p)
                continue

            for f in faces:
                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    continue
                vec = emb.astype(np.float32)
                vec /= np.linalg.norm(vec)
                embeddings.append(vec)
                owners.append(p)
    finally:
        sys.stdout = old_stdout

    if not embeddings:
        return {
            "clusters": {},
            "plan": [],
            "unreadable": [str(p) for p in unreadable],
            "no_faces": [str(p) for p in no_faces],
            "tqdm_log": tqdm_output.getvalue(),
        }

    X = np.vstack(embeddings)
    model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
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
    for path in sorted(all_images, key=lambda x: str(x)):
        clusters = cluster_by_img.get(path)
        if not clusters:
            continue
        plan.append({
            "path": str(path),
            "cluster": sorted(list(clusters)),
        })

    return {
        "clusters": {int(k): [str(p) for p in sorted(v, key=lambda x: str(x))] for k, v in cluster_map.items()},
        "plan": plan,
        "unreadable": [str(p) for p in unreadable],
        "no_faces": [str(p) for p in no_faces],
        "tqdm_log": tqdm_output.getvalue(),
    }
