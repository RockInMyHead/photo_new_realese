import streamlit as st
import json
import shutil
from pathlib import Path
from core.cluster import build_plan

st.set_page_config("Кластеризация лиц", layout="wide")
st.title("📸 Кластеризация лиц и распределение по папкам")

photo_dir = st.text_input("Путь к папке с изображениями:", value="C:/test/m4")

def distribute_to_folders(plan, base_dir: Path):
    moved, copied = 0, 0
    for item in plan.get("plan", []):
        src = Path(item["path"])
        clusters = item["cluster"]

        if not src.exists():
            continue

        if len(clusters) == 1:
            dst_folder = base_dir / f"cluster_{clusters[0]}"
            dst_folder.mkdir(exist_ok=True)
            dst = dst_folder / src.name
            try:
                shutil.move(str(src), str(dst))
                moved += 1
            except Exception as e:
                st.error(f"❌ Ошибка перемещения {src} → {dst}: {e}")
        else:
            for cluster_id in clusters:
                dst_folder = base_dir / f"cluster_{cluster_id}"
                dst_folder.mkdir(exist_ok=True)
                dst = dst_folder / src.name
                try:
                    shutil.copy2(str(src), str(dst))
                    copied += 1
                except Exception as e:
                    st.error(f"❌ Ошибка копирования {src} → {dst}: {e}")
    return moved, copied

if st.button("🔍 Запустить кластеризацию и распределение"):
    path = Path(photo_dir)
    if not path.exists():
        st.error("Путь не существует.")
    else:
        with st.spinner("Кластеризация..."):
            plan = build_plan(path)
        with open("plan.json", "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

        moved, copied = distribute_to_folders(plan, path)

        st.success(f"✅ Готово! Перемещено: {moved}, Скопировано (многокластерные): {copied}")
        if plan.get("unreadable"):
            st.warning(f"Нечитаемых файлов: {len(plan['unreadable'])}")
            st.code("\n".join(plan["unreadable"][:30]))

        st.subheader("Примеры кластеров:")
        for entry in plan.get("plan", [])[:10]:
            st.json(entry)