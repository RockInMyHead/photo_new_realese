import streamlit as st
import json
from pathlib import Path
from core.cluster import build_plan

st.set_page_config("Кластеризация лиц", layout="wide")
st.title("📸 Кластеризация лиц с поддержкой кириллических путей")

photo_dir = st.text_input("Путь к папке с изображениями:", value="C:/test/m4")

if st.button("🔍 Запустить кластеризацию"):
    path = Path(photo_dir)
    if not path.exists():
        st.error("Путь не существует.")
    else:
        with st.spinner("Работаю..."):
            plan = build_plan(path)
        with open("plan.json", "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

        st.success(f"Готово! Кластеров: {len(plan.get('clusters', {}))}")
        if plan.get("unreadable"):
            st.warning(f"Нечитаемых файлов: {len(plan['unreadable'])}")
            st.code("\n".join(plan["unreadable"][:30]))

        st.subheader("Примеры кластеров:")
        for entry in plan.get("plan", [])[:10]:
            st.json(entry)