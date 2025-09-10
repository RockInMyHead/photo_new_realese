
import streamlit as st
import json
from pathlib import Path
from core.cluster import build_plan

st.title("📸 Кластеризация лиц (InsightFace + DBSCAN)")

photos_dir = st.text_input("Папка с фото:", value="C:/test/m3")

if st.button("🔍 Кластеризовать лица"):
    path = Path(photos_dir)
    if not path.exists():
        st.error("Путь не существует.")
    else:
        with st.spinner("Выполняется кластеризация..."):
            plan = build_plan(path)

        cluster_map = plan.get("clusters", {})
        unreadable = plan.get("unreadable", [])
        result_plan = plan.get("plan", [])

        with open("plan.json", "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

        st.success(f"✅ Кластеризация завершена. Обнаружено кластеров: {len(cluster_map)}.")

        if unreadable:
            st.warning(f"⚠️ Не удалось прочитать {len(unreadable)} файл(ов).")
            st.code("\n".join(unreadable[:50]), language="text")

        st.write("Примеры кластеризации:")
        for row in result_plan[:5]:
            st.json(row)
