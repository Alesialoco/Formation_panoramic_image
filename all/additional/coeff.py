import cv2
import numpy as np

def chessboard_calibration(video_path, chessboard_size, square_width, square_height, output_path=None):    
    """
    Калибровка камеры с использованием шахматной доски
    
    Args:
        video_path (str): Путь к видеофайлу с шахматной доской
        chessboard_size (tuple): Размер шахматной доски (ширина, высота) в клетках
        square_width (float): Ширина одной клетки в мм
        square_height (float): Высота одной клетки в мм
        output_path (str, optional): Путь для сохранения результатов калибровки
        
    Returns:
        tuple or None: Матрица камеры, коэффициенты дисторсии и ошибка репроекции,
                       или None в случае ошибки
    """
    # Критерии для субпиксельного уточнения углов
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Создание массива 3D точек для шахматной доски
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    x_coords = np.arange(0, chessboard_size[0]) * square_width
    y_coords = np.arange(0, chessboard_size[1]) * square_height
    xx, yy = np.meshgrid(x_coords, y_coords)
    objp[:, :2] = np.column_stack([xx.ravel(), yy.ravel()])
    objpoints = []  # 3D точки в реальном мире
    imgpoints = []  # 2D точки на изображении
    
    # Открытие видеофайла
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео файл")
        return
    
    frame_count = 0
    successful_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        
        if ret:
            # Уточнение позиции углов с субпиксельной точностью
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            successful_frames += 1
            
            # Визуализация найденных углов
            cv2.drawChessboardCorners(frame, chessboard_size, corners_refined, ret)
            cv2.putText(frame, f"Found: {successful_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Corners: {chessboard_size[0]}x{chessboard_size[1]}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if frame_count % 100 == 0:
            print(f"Обработано кадров: {frame_count}, Найдено досок: {successful_frames}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nВсего обработано кадров: {frame_count}. Успешно найдено досок: {successful_frames}")

    # Проверка достаточности данных для калибровки
    if successful_frames < 10:
        print("Ошибка: Недостаточно кадров с шахматной доской для калибровки (минимум 10)")
        return None
    
    print("\nВыполняется калибровка камеры...")
    # Калибровка камеры
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Вычисление ошибки репроекции
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)
    
    # Сохранение результатов калибровки
    if output_path:
        calibration_data = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'reprojection_error': mean_error
        }
        np.savez(output_path, **calibration_data)
        print(f"\nРезультаты сохранены в: {output_path}.npz")
    
    return camera_matrix, dist_coeffs, mean_error


def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Удаление дисторсии с изображения
    
    Args:
        image (numpy.ndarray): Искаженное изображение
        camera_matrix (numpy.ndarray): Матрица камеры
        dist_coeffs (numpy.ndarray): Коэффициенты дисторсии
        
    Returns:
        tuple: Исправленное изображение и новая матрица камеры
    """
    h, w = image.shape[:2]
    # Получение оптимальной новой матрицы камеры и ROI
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    # Удаление дисторсии
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    # Обрезка по ROI
    undistorted = undistorted[y:y+h, x:x+w]
    return undistorted, new_camera_matrix


def test_calibration(video_path, camera_matrix, dist_coeffs, output_video_path=None):
    """
    Тестирование калибровки на видео
    
    Args:
        video_path (str): Путь к тестовому видео
        camera_matrix (numpy.ndarray): Матрица камеры
        dist_coeffs (numpy.ndarray): Коэффициенты дисторсии
        output_video_path (str, optional): Путь для сохранения исправленного видео
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео файл")
        return
    
    # Получение параметров видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Инициализация VideoWriter для сохранения результата
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        # Удаление дисторсии с кадра
        undistorted_frame, _ = undistort_image(frame, camera_matrix, dist_coeffs)
        undistorted_resized = cv2.resize(undistorted_frame, (width, height))
        
        # Сохранение кадра
        if output_video_path:
            out.write(undistorted_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if output_video_path:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """Основная функция для запуска калибровки и тестирования"""
    # Параметры шахматной доски
    CHESSBOARD_SIZE = (7, 7) 
    SQUARE_WIDTH = 49.5  # мм
    SQUARE_HEIGHT = 49.7  # мм
    
    # Пути к файлам
    VIDEO_PATH = "Chessboard_first_1500_frames.avi"
    OUTPUT_CALIBRATION = "calibration_results"
    OUTPUT_VIDEO = "dist.mp4"
    
    print("КАЛИБРОВКА КАМЕРЫ С ШАХМАТНОЙ ДОСКОЙ")
    
    # Загрузка предварительно сохраненных результатов калибровки
    data = np.load('calibration_results.npz')
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    reprojection_error = data['reprojection_error']
    
    # Тестирование калибровки на видео
    test_calibration(
            video_path=VIDEO_PATH,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            output_video_path=OUTPUT_VIDEO
    )
