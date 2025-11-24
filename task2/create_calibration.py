import cv2
import numpy as np
import glob
import pickle
import os

def calibrate_camera(images_path="chess_desks/*.jpg", output_file="camera_calibration.pkl", pattern_size=(9,6), square_size=25.0):
    image_files = glob.glob(images_path)
    if len(image_files) == 0:
        print(f"Изображения не найдены по пути: {images_path}")
        return False

    print(f"Найдено {len(image_files)} изображений в папке проекта")

    images = []
    for fname in image_files:
        img = cv2.imread(fname)
        if img is not None:
            images.append(img)
            print(f"Загружено: {os.path.basename(fname)}")
        else:
            print(f"Ошибка загрузки: {os.path.basename(fname)}")

    if len(images) < 2:
        print(f'Слишком мало изображений! Найдено {len(images)} изображений')
        return False

    print(f"Загружено {len(images)} изображений для калибровки")

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2) * square_size

    objpoints = []
    imgpoints = []

    print("Поиск шахматных досок на изображениях...")

    successful_calibrations = 0
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners_refined)
            successful_calibrations += 1

            print(f"Шахматная доска найдена: {os.path.basename(image_files[i])}")
        else:
            print(f"Шахматная доска не найдена: {os.path.basename(image_files[i])}")

    cv2.destroyAllWindows()

    print(f"Найдено {successful_calibrations} подходящих изображений")
    print("Выполняется калибровка камеры...")

    try:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, images[0].shape[1::-1], None, None)
        data = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'calibration_info': {
                'images_used': successful_calibrations,
                'total_images': len(images),
                'reprojection_error': ret,
                'pattern_size': pattern_size,
                'square_size': square_size
            }
        }

        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        return True
    except Exception as e:
        print(f"Ошибка при калибровке: {e}")
        return False

def main():
    # Параметры по умолчанию (все в одной папке проекта)
    IMAGES_PATH = "chess_desks/*.jpg"
    OUTPUT_FILE = "camera_calibration.pkl"
    PATTERN_SIZE = (9, 6)
    SQUARE_SIZE = 25.0

    if not glob.glob(IMAGES_PATH):
        print(f"Папка 'chess_desks' не найдена в папке проекта!")
        return

    success = calibrate_camera(
        images_path=IMAGES_PATH,
        output_file=OUTPUT_FILE,
        pattern_size=PATTERN_SIZE,
        square_size=SQUARE_SIZE
    )

    if success:
        print("\nКалибровка завершена успешно!")
    else:
        print("\nКалибровка не удалась!")

if __name__ == '__main__':
    main()
