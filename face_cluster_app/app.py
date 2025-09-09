import streamlit as st
from pathlib import Path
import json
import shutil
from core.cluster import build_plan, IMG_EXTS

def load_config() -> dict:
    return {
        "group_thr": 3,
        "eps_sim": 0.72,
        "min_samples": 2,
        "min_face": 60,
        "blur_thr": 60.0,
        "det_size": 640,
        "gpu_id": 0,
    }

CFG = load_config()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_copy(src: Path, dst_dir: Path):
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if dst.exists():
        stem, ext = src.stem, src.suffix
        i = 1
        while (dst_dir / f"{stem}_{i}{ext}").exists():
            i += 1
        dst = dst_dir / f"{stem}_{i}{ext}"
    shutil.copy2(src, dst)

st.set_page_config(page_title="Face Cluster Tool", layout="wide")
st.title("👥 Face Clustering App (InsightFace + DBSCAN)")

folder = st.text_input("Путь к папке с изображениями", value="", placeholder="Введите путь...")

if folder:
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        st.error("Указанный путь не существует или не является директорией.")
    else:
        if st.button("▶️ Начать кластеризацию", type="primary"):
            with st.spinner("Обработка изображений..."):
                plan = build_plan(
                    group_dir=folder_path,
                    group_thr=CFG["group_thr"],
                    eps_sim=CFG["eps_sim"],
                    min_samples=CFG["min_samples"],
                    min_face=CFG["min_face"],
                    blur_thr=CFG["blur_thr"],
                    det_size=CFG["det_size"],
                    gpu_id=CFG["gpu_id"]
                )

            st.success("Кластеризация завершена.")
            st.subheader("📊 Сводка:")
            st.write(f"Всего изображений: {plan['stats']['images_total']}")
            st.write(f"Неизвестные изображения: {plan['stats']['images_unknown_only']}")
            st.write(f"Групповые изображения: {plan['stats']['images_group_only']}")
            st.write(f"Найдено кластеров: {len(plan['eligible_clusters'])}")

            st.info("📁 Распределение изображений по папкам...")
            output_root = folder_path / "clusters_output"
            shutil.rmtree(output_root, ignore_errors=True)
            ensure_dir(output_root)

            for cid in plan["eligible_clusters"]:
                img_paths = plan["cluster_images"].get(cid, [])
                cluster_dir = output_root / f"cluster_{cid}"
                for img in img_paths:
                    safe_copy(Path(img), cluster_dir)

            group_dir = output_root / "__group_only__"
            for img in plan["group_only_images"]:
                safe_copy(Path(img), group_dir)

            unknown_dir = output_root / "__unknown__"
            for img in plan["unknown_images"]:
                safe_copy(Path(img), unknown_dir)

            st.success(f"Файлы распределены по директории: `{output_root}`")

            st.subheader("🖼️ Примеры по кластерам:")
            for cid in plan["eligible_clusters"]:
                imgs = plan["cluster_images"].get(cid, [])[:5]
                st.markdown(f"**Кластер {cid}** ({len(plan['cluster_images'][cid])} фото):")
                st.image(imgs, width=150)

            result_path = output_root / "face_clusters.json"
            with result_path.open("w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
            st.success(f"Результаты сохранены в: `{result_path}`")