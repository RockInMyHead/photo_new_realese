
import streamlit as st
from core.cluster import build_plan, IMG_EXTS
from pathlib import Path
import shutil
import json
import numpy as np

st.title("Face Clustering App")

input_dir = st.text_input("📂 Путь к папке с фото", value="photos")
run = st.button("🔍 Кластеризовать лица")

if run:
    path = Path(input_dir)
    if not path.exists():
        st.error("Папка не найдена.")
        st.stop()

    st.write("📦 Сканирование изображений...")
    plan = build_plan(path)

    clustered_paths = set()
    cluster_map = {}  # кластер -> список фото
    shared_files = set()

    for item in plan:
        if isinstance(item["cluster"], list):  # общее фото
            shared_files.add(item["path"])
            for cluster in item["cluster"]:
                cluster_map.setdefault(cluster, []).append(item["path"])
        else:
            cluster_map.setdefault(item["cluster"], []).append(item["path"])
            clustered_paths.add(item["path"].resolve())

    st.write("📁 Размещение по кластерам...")
    for cluster, paths in cluster_map.items():
        dst_dir = path / f"cluster_{cluster}"
        dst_dir.mkdir(exist_ok=True)
        for src_path in paths:
            dst_path = dst_dir / src_path.name
            try:
                shutil.copy(src_path, dst_path)
            except Exception as e:
                st.warning(f"Ошибка копирования {src_path}: {e}")

    st.write("🧹 Удаление исходных одиночных фото...")
    for p in clustered_paths:
        try:
            p.unlink()
        except Exception as e:
            st.warning(f"Не удалось удалить {p.name}: {e}")

    # Преобразование в сериализуемый формат
    def convert_json_safe(obj):
        if isinstance(obj, dict):
            return {k: convert_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_json_safe(i) for i in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray, set, tuple)):
            return [convert_json_safe(i) for i in obj]
        else:
            return obj

    with open("plan.json", "w", encoding="utf-8") as f:
        json.dump(convert_json_safe(plan), f, ensure_ascii=False, indent=2)

    st.success(f"✅ Завершено! Найдено кластеров: {len(cluster_map)}")
