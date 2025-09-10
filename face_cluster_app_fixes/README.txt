Установка:
    pip install insightface onnxruntime opencv-python-headless scikit-learn tqdm numpy

Пример:
    from core.cluster import build_plan
    result = build_plan("C:/путь/к/папке/с/фото")

Если фото не читаются — убедись, что они реально существуют и передаются с Unicode-путями.