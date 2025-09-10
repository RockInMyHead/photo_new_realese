
from insightface.app import FaceAnalysis
import numpy as np
from pathlib import Path
import cv2
from sklearn.cluster import DBSCAN
from tqdm import tqdm

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def is_image(p: Path):
    return p.suffix.lower() in IMG_EXTS

def build_plan(input_dir: Path, det_size=(1024, 1024), dbscan_eps=0.5, dbscan_min_samples=2):
    input_dir = Path(input_dir)
    all_images = [p for p in input_dir.rglob("*") if is_image(p)]

    model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=0, det_size=det_size)

    embeddings = []
    img_index = []
    for img_path in tqdm(all_images, desc="📷 Scanning"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        faces = model.get(img)
        for face in faces:
            embeddings.append(face.embedding)
            img_index.append(img_path)

    if not embeddings:
        return []

    embeddings = np.array(embeddings)
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine").fit(embeddings)
    labels = clustering.labels_

    cluster_by_img = {}  # path -> list of clusters
    for path, label in zip(img_index, labels):
        if label == -1:
            continue  # нераспознанные
        cluster_by_img.setdefault(path, set()).add(label)

    plan = []
    for path in all_images:
        clusters = cluster_by_img.get(path)
        if not clusters:
            continue
        if len(clusters) == 1:
            plan.append({"path": path, "cluster": list(clusters)[0]})
        else:
            plan.append({"path": path, "cluster": list(clusters)})

    return plan
