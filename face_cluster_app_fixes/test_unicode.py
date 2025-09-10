from core.cluster import imread_safe
from pathlib import Path
import cv2

# Заменить на путь с кириллицей на вашей машине
img_path = Path(r"C:/test/m4/Фото27.jpg")

img = imread_safe(img_path)
if img is not None:
    print("✅ Файл прочитан успешно!")
    cv2.imwrite("output.jpg", img)
else:
    print("❌ Ошибка чтения файла.")