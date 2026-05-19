import cv2
import numpy as np
from coeff import chessboard_calibration

# Параметры шахматной доски
CHESSBOARD_SIZE = (7, 7)      # Количество внутренних углов (7x7 = 49 углов)
SQUARE_WIDTH = 49.5           # Ширина клетки в мм
SQUARE_HEIGHT = 49.7          # Высота клетки в мм

# Пути к файлам
VIDEO_PATH = "Chess_trimmed.avi"    # Ваше видео с шахматной доской
OUTPUT_CALIBRATION = "calibration_results_2.npz"   # Файл для сохранения параметров

print("=== НАЧАЛО КАЛИБРОВКИ КАМЕРЫ ===")
print(f"Параметры доски: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} углов")
print(f"Размер клетки: {SQUARE_WIDTH} x {SQUARE_HEIGHT} мм")
print(f"Видео: {VIDEO_PATH}")
print()

# Выполнение калибровки и сохранение результатов
result = chessboard_calibration(
    video_path=VIDEO_PATH,
    chessboard_size=CHESSBOARD_SIZE,
    square_width=SQUARE_WIDTH,
    square_height=SQUARE_HEIGHT,
    output_path=OUTPUT_CALIBRATION.replace('.npz', '')  # без расширения
)

if result is not None:
    camera_matrix, dist_coeffs, reprojection_error = result
    
    print("\n=== РЕЗУЛЬТАТЫ КАЛИБРОВКИ ===")
    print(f"Матрица камеры:\n{camera_matrix}")
    print(f"\nКоэффициенты дисторсии:\n{dist_coeffs}")
    print(f"\nОшибка репроекции: {reprojection_error:.4f} пикселей")
    print(f"\nПараметры сохранены в: {OUTPUT_CALIBRATION}")
    
    # Дополнительная информация
    if reprojection_error < 0.5:
        print("✓ Отличная калибровка!")
    elif reprojection_error < 1.0:
        print("✓ Хорошая калибровка")
    else:
        print("⚠ Калибровка может быть неточной. Попробуйте использовать больше кадров.")
else:
    print("Ошибка: Не удалось выполнить калибровку.")
    print("Убедитесь, что:")
    print("1. Видео файл существует")
    print("2. Шахматная доска хорошо видна в видео")
    print("3. Размеры доски (количество углов) указаны правильно")
    print("4. В видео найдено минимум 10 кадров с доской")
