import cv2
import numpy as np
from pathlib import Path
from typing import Dict
from collections import defaultdict
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def build_plan(
    group_dir: Path,
    group_thr: int = 3,
    eps_sim: float = 0.72,
    min_samples: int = 2,
    min_face: int = 60,
    blur_thr: float = 60.0,
    det_size: int = 640,
    gpu_id: int = 0
) -> Dict:
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=gpu_id, det_size=(det_size, det_size))

    image_paths = [
        p for p in group_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()
    ]

    embeddings = []
    image_map = []
    stats = {
        "images_total": len(image_paths),
        "images_unknown_only": 0,
        "images_group_only": 0
    }

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        for face in faces:
            if face.det_score < 0.7:
                continue
            if face.bbox[2] - face.bbox[0] < min_face or face.bbox[3] - face.bbox[1] < min_face:
                continue
            embeddings.append(face.embedding)
            image_map.append((img_path, face.embedding))

    if not embeddings:
        stats["images_unknown_only"] = len(image_paths)
        return {
            "eligible_clusters": [],
            "cluster_centroids": {},
            "cluster_images": {},
            "group_only_images": [],
            "unknown_images": image_paths,
            "stats": stats,
        }

    emb_array = np.stack(embeddings)
    emb_norm = emb_array / (np.linalg.norm(emb_array, axis=1, keepdims=True) + 1e-12)
    clustering = DBSCAN(eps=(1 - eps_sim), min_samples=min_samples, metric="cosine").fit(emb_norm)
    labels = clustering.labels_

    clusters = defaultdict(list)
    for idx, lbl in enumerate(labels):
        if lbl == -1:
            continue
        clusters[lbl].append(image_map[idx][0])

    cluster_centroids = {}
    cluster_images = {}
    eligible_clusters = []
    for cid, paths in clusters.items():
        if len(paths) < group_thr:
            continue
        eligible_clusters.append(cid)
        embs = [image_map[i][1] for i, l in enumerate(labels) if l == cid]
        mean_emb = np.mean(embs, axis=0)
        norm_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
        cluster_centroids[cid] = norm_emb.tolist()
        cluster_images[cid] = [str(p) for p in paths]

    joint_freq = defaultdict(int)
    for lbl, (p, _) in zip(labels, image_map):
        if lbl != -1:
            joint_freq[str(p)] += 1

    group_only_images = [str(p) for p, c in joint_freq.items() if c > 1]
    unknown_images = [str(p) for i, (p, _) in enumerate(image_map) if labels[i] == -1]

    stats["images_unknown_only"] = len(unknown_images)
    stats["images_group_only"] = len(group_only_images)

    return {
        "eligible_clusters": eligible_clusters,
        "cluster_centroids": cluster_centroids,
        "cluster_images": cluster_images,
        "group_only_images": group_only_images,
        "unknown_images": unknown_images,
        "stats": stats,
    }