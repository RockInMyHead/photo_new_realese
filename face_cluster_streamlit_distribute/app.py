import streamlit as st
import json
import shutil
from pathlib import Path
from PIL import Image
import psutil
from core.cluster import build_plan, IMG_EXTS

st.set_page_config("Кластеризация лиц", layout="wide")
st.title("📸 Кластеризация лиц и распределение по папкам")

if "queue" not in st.session_state:
    st.session_state["queue"] = []

def get_logical_drives():
    return [Path(p.mountpoint) for p in psutil.disk_partitions(all=False) if Path(p.mountpoint).exists()]

def get_special_dirs():
    home = Path.home()
    return {
        "💼 Рабочий стол": home / "Desktop",
        "📄 Документы": home / "Documents",
        "📥 Загрузки": home / "Downloads",
        "🖼 Изображения": home / "Pictures",
    }

def show_folder_contents(current_path: Path):
    st.markdown(f"📁 **Текущая папка:** `{current_path}`")

    if st.button("📌 Добавить в очередь", key=f"queue_{current_path}"):
        if str(current_path) not in st.session_state["queue"]:
            st.session_state["queue"].append(str(current_path))
            st.success(f"Добавлено в очередь: {current_path}")

    if current_path.parent != current_path:
        if st.button("⬅️ Назад", key=f"up_{current_path}"):
            st.session_state["current_path"] = str(current_path.parent)
            st.rerun()

    try:
        images = [f for f in current_path.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
        if images:
            st.markdown("### 🖼 Изображения:")
            for img_path in images[:10]:
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB").resize((100, 100))
                        st.image(img, caption=img_path.name, width=100)
                except Exception:
                    st.warning(f"⚠️ Ошибка при отображении: {img_path.name}")
    except Exception:
        st.warning("⚠️ Ошибка доступа к файлам или изображениям")

    st.markdown("---")

    try:
        subdirs = sorted([p for p in current_path.iterdir() if p.is_dir()], key=lambda x: x.name.lower())
        for folder in subdirs:
            if st.button(f"📂 {folder.name}", key=f"enter_{folder}"):
                st.session_state["current_path"] = str(folder)
                st.rerun()
    except PermissionError:
        st.error(f"⛔ Нет доступа к содержимому: `{current_path}`")
    except Exception as e:
        st.error(f"❌ Ошибка при обходе `{current_path}`: {e}")

if "current_path" not in st.session_state:
    roots = get_logical_drives() + list(get_special_dirs().values())
    for root in roots:
        if root.exists():
            st.session_state["current_path"] = str(root)
            break
    else:
        st.session_state["current_path"] = str(Path.home())

st.subheader("🧭 Переход по дискам и системным папкам")

cols = st.columns(4)
for i, drive in enumerate(get_logical_drives()):
    with cols[i % 4]:
        if st.button(f"💽 {drive}", key=f"drive_{drive}"):
            st.session_state["current_path"] = str(drive)
            st.rerun()

for name, path in get_special_dirs().items():
    if path.exists():
        if st.button(name, key=f"special_{name}"):
            st.session_state["current_path"] = str(path)
            st.rerun()

st.markdown("""
    <style>
    .folder-scroll-box {
        max-height: 700px;
        overflow-y: auto;
        padding: 0 10px;
        border: 1px solid #ccc;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    </style>
    <div class='folder-scroll-box'>
""", unsafe_allow_html=True)

show_folder_contents(Path(st.session_state["current_path"]))

st.markdown("</div>", unsafe_allow_html=True)

if st.session_state["queue"]:
    st.subheader("📋 Очередь на обработку:")
    for i, folder in enumerate(st.session_state["queue"]):
        st.text(f"{i+1}. {folder}")
    if st.button("🧹 Очистить очередь"):
        st.session_state["queue"] = []

# --- Распределение по кластерам ---
def distribute_to_folders(plan, base_dir: Path):
    moved, copied = 0, 0
    cluster_dirs = set()
    moved_paths = set()

    for item in plan.get("plan", []):
        src = Path(item["path"])
        clusters = item["cluster"]
        if not src.exists():
            continue

        if len(clusters) == 1:
            cluster_id = clusters[0]
            dst = base_dir / f"cluster_{cluster_id}" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(src), str(dst))
                moved += 1
                cluster_dirs.add(cluster_id)
                moved_paths.add(src.parent)
            except Exception as e:
                st.error(f"❌ Ошибка перемещения {src} → {dst}: {e}")
        else:
            success_count = 0
            total_targets = 0
            for cluster_id in clusters:
                dst = base_dir / f"cluster_{cluster_id}" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                total_targets += 1
                try:
                    shutil.copy2(str(src), str(dst))
                    copied += 1
                    success_count += 1
                except Exception as e:
                    st.error(f"❌ Ошибка копирования {src} → {dst}: {e}")
            if success_count == total_targets:
                try:
                    src.unlink()
                    moved_paths.add(src.parent)
                except Exception as e:
                    st.warning(f"⚠️ Не удалось удалить оригинал {src}: {e}")

    for p in sorted(moved_paths, key=lambda x: len(str(x)), reverse=True):
        try:
            if p.exists() and not any(p.iterdir()):
                p.rmdir()
        except Exception:
            pass

    return moved, copied

# --- Обработка очереди ---
if st.session_state["queue"] and st.button("🚀 Обработать всю очередь"):
    cluster_offset = 1  # глобальный счётчик кластеров

    for folder in st.session_state["queue"]:
        path = Path(folder)
        st.markdown(f"### 📂 Обработка: `{path}`")
        if not path.exists():
            st.error("❌ Путь не существует.")
            continue

        with st.spinner("🧠 Кластеризация..."):
            plan = build_plan(path)

        # --- Перенумерация кластеров ---
        old_to_new = {}
        for i, cid in enumerate(sorted(plan.get("clusters", {}).keys()), start=cluster_offset):
            old_to_new[int(cid)] = i

        plan["clusters"] = {
            old_to_new[int(k)]: v for k, v in plan.get("clusters", {}).items() if int(k) in old_to_new
        }

        for entry in plan.get("plan", []):
            entry["cluster"] = [old_to_new[cid] for cid in entry.get("cluster", []) if cid in old_to_new]

        cluster_count = len(old_to_new)
        cluster_offset += cluster_count

        with open(f"plan_{path.name}.json", "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

        moved, copied = distribute_to_folders(plan, path)

        st.success(f"✅ Готово. Перемещено: {moved}, Скопировано и удалено оригиналов: {copied}")

        if plan.get("unreadable"):
            st.warning(f"📛 Нечитаемых файлов: {len(plan['unreadable'])}")
            st.code("\n".join(plan["unreadable"][:30]))

        if plan.get("no_faces"):
            st.warning(f"🙈 Без лиц: {len(plan['no_faces'])}")
            st.code("\n".join(plan["no_faces"][:30]))

        if plan.get("tqdm_log"):
            st.code(plan["tqdm_log"], language="bash")

    st.session_state["queue"] = []
