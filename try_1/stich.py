import cv2
import numpy as np
from typing import Tuple, List, Optional
import os
import math
import logging
import sys


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class OptimizedCylindricalStitcher:
    """
    Класс для сшивки двух видео с цилиндрической проекцией
    """
    
    def __init__(self, video1_path: str, video2_path: str, output_path: str,
                 num_calibration_frames: int = 10, neutral_plane_t: float = 0.5,
                 fov_horizontal: float = 150):
        """
        Инициализация класса для сшивки видео
        
        Args:
            video1_path: Путь к первому видео
            video2_path: Путь ко второму видео
            output_path: Путь для сохранения результата
            num_calibration_frames: Количество кадров для калибровки гомографии
            neutral_plane_t: Параметр нейтральной плоскости (0-1)
            fov_horizontal: Горизонтальный угол обзора для цилиндрической проекции
            
        Raises:
            ValueError: Если neutral_plane_t не в диапазоне [0, 1]
        """
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.output_path = output_path
        self.num_calibration_frames = num_calibration_frames
        self.neutral_plane_t = neutral_plane_t
        self.fov_horizontal = fov_horizontal

        if not 0 <= neutral_plane_t <= 1:
            logger.error(f"neutral_plane_t должен быть в диапазоне от 0 до 1, получено: {neutral_plane_t}")
            raise ValueError("neutral_plane_t должен быть в диапазоне от 0 до 1")

        # Инициализация детектора и сопоставителя признаков
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Параметры сшивки
        self.homography_matrix_1_to_2 = None
        self.homography_matrix_2_to_1 = None
        self.neutral_transform_2 = None
        self.neutral_transform_2_corrected = None
        self.new_homography_1_to_2_neutral = None
        self.final_transform_1 = None
        self.final_transform_2 = None
        self.output_size = None
        self.blend_mask = None
        self.blend_mask_3d = None

        # Параметры цилиндрической проекции и обрезки
        self.cylindrical_map_x = None
        self.cylindrical_map_y = None
        self.horizontal_crop_slices = None
        self.final_crop_slices = None
        self.final_output_size = None
        self.horizontal_crop_percent = 0.30

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Извлечение признаков SIFT из изображения
        
        Args:
            image: Входное изображение в формате BGR
            
        Returns:
            Кортеж (keypoints, descriptors):
                keypoints: Список ключевых точек
                descriptors: Дескрипторы ключевых точек
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def find_homography(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        """
        Поиск матрицы гомографии между двумя изображениями
        
        Args:
            img1: Первое изображение
            img2: Второе изображение
            
        Returns:
            Матрица гомографии 3x3 или None, если не удалось найти
        """
        kp1, desc1 = self.extract_features(img1)
        kp2, desc2 = self.extract_features(img2)

        if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
            logger.warning("Недостаточно дескрипторов для поиска гомографии")
            return None

        matches = self.flann.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            logger.warning(f"Недостаточно хороших совпадений: {len(good_matches)}")
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def calculate_homography_from_frames(self) -> None:
        """
        Вычисление матриц гомографии на основе нескольких кадров из видео
        
        Raises:
            Exception: Если не удалось вычислить матрицы гомографии
        """
        logger.info("Вычисление матриц гомографии...")
        
        cap1 = cv2.VideoCapture(self.video1_path)
        cap2 = cv2.VideoCapture(self.video2_path)

        homographies_1_to_2 = []
        homographies_2_to_1 = []

        for i in range(self.num_calibration_frames):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            H_1_to_2 = self.find_homography(frame1, frame2)
            H_2_to_1 = self.find_homography(frame2, frame1)

            if H_1_to_2 is not None and H_2_to_1 is not None:
                homographies_1_to_2.append(H_1_to_2)
                homographies_2_to_1.append(H_2_to_1)

        cap1.release()
        cap2.release()

        if homographies_1_to_2 and homographies_2_to_1:
            self.homography_matrix_1_to_2 = np.median(homographies_1_to_2, axis=0)
            self.homography_matrix_2_to_1 = np.median(homographies_2_to_1, axis=0)
            logger.info(f"Матрицы гомографии вычислены по {len(homographies_1_to_2)} кадрам")
        else:
            logger.error("Не удалось вычислить матрицы гомографии")
            raise Exception("Не удалось вычислить матрицы гомографии")

    def neutral_plane_transform(self) -> np.ndarray:
        """
        Создание преобразования для нейтральной плоскости
        
        Returns:
            Матрица преобразования 3x3 для нейтральной плоскости
        """
        logger.info(f"Создание преобразования для нейтральной плоскости (t={self.neutral_plane_t})...")

        H2_to_1 = self.homography_matrix_2_to_1 / self.homography_matrix_2_to_1[2, 2]

        H2_neutral = (np.eye(3) * (1 - self.neutral_plane_t) + H2_to_1 * self.neutral_plane_t)
        H2_neutral = H2_neutral / H2_neutral[2, 2]

        return H2_neutral

    def calculate_new_homography_to_neutral_plane(self, frame1: np.ndarray, frame2: np.ndarray) -> None:
        """
        Вычисление новой гомографии к нейтральной плоскости
        
        Args:
            frame1: Первый кадр
            frame2: Второй кадр
            
        Raises:
            Exception: Если не удалось найти контент или вычислить гомографию
        """
        logger.info("Вычисление новой гомографии для нейтральной плоскости...")

        h2, w2 = frame2.shape[:2]
        warped2_neutral = cv2.warpPerspective(frame2, self.neutral_transform_2, (w2 * 2, h2 * 2),
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        gray_warped = cv2.cvtColor(warped2_neutral, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_warped, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.error("Не удалось найти контент трансформированного видео2")
            raise Exception("Не удалось найти контент трансформированного видео2")

        all_points = np.vstack([contour for contour in contours])
        x, y, w, h = cv2.boundingRect(all_points)

        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(warped2_neutral.shape[1] - x, w + 2 * margin)
        h = min(warped2_neutral.shape[0] - y, h + 2 * margin)

        warped2_cropped = warped2_neutral[y:y+h, x:x+w]

        correction_matrix = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
        self.neutral_transform_2_corrected = correction_matrix @ self.neutral_transform_2

        self.new_homography_1_to_2_neutral = self.find_homography(frame1, warped2_cropped)

        if self.new_homography_1_to_2_neutral is None:
            logger.error("Не удалось вычислить новую гомографии")
            raise Exception("Не удалось вычислить новую гомографии")

    def calculate_final_transforms(self, frame1: np.ndarray, frame2: np.ndarray) -> None:
        """
        Вычисление финальных трансформаций для сшивки
        
        Args:
            frame1: Первый кадр
            frame2: Второй кадр
        """
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

        corners1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
        corners1_transformed = cv2.perspectiveTransform(corners1.reshape(-1, 1, 2),
                                                      self.new_homography_1_to_2_neutral)

        corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
        corners2_transformed = cv2.perspectiveTransform(corners2.reshape(-1, 1, 2),
                                                      self.neutral_transform_2_corrected)

        all_corners = np.vstack([corners1_transformed, corners2_transformed])
        min_x, min_y = np.min(all_corners[:, 0, :], axis=0)
        max_x, max_y = np.max(all_corners[:, 0, :], axis=0)

        padding_x = 80
        padding_y = 80
        total_width = int(max_x - min_x) + 2 * padding_x
        total_height = int(max_y - min_y) + 2 * padding_y

        translation = np.array([[1, 0, -min_x + padding_x],
                              [0, 1, -min_y + padding_y],
                              [0, 0, 1]])

        self.final_transform_1 = translation @ self.new_homography_1_to_2_neutral
        self.final_transform_2 = translation @ self.neutral_transform_2_corrected
        self.output_size = (total_width, total_height)

        logger.info(f"Размер панорамы: {total_width}x{total_height}")

    def precompute_blend_mask(self, frame1: np.ndarray, frame2: np.ndarray) -> None:
        """
        Предварительное вычисление маски для плавного смешивания
        
        Args:
            frame1: Первый кадр
            frame2: Второй кадр
        """
        warped1 = cv2.warpPerspective(frame1, self.final_transform_1, self.output_size,
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped2 = cv2.warpPerspective(frame2, self.final_transform_2, self.output_size,
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        mask1 = (warped1.sum(axis=2) > 10)
        mask2 = (warped2.sum(axis=2) > 10)
        overlap = mask1 & mask2

        if not np.any(overlap):
            blend_start = self.output_size[0] // 2 - 150
            blend_end = self.output_size[0] // 2 + 150
        else:
            overlap_cols = np.where(np.any(overlap, axis=0))[0]
            if len(overlap_cols) == 0:
                blend_start = self.output_size[0] // 2 - 150
                blend_end = self.output_size[0] // 2 + 150
            else:
                blend_start = overlap_cols[0]
                blend_end = overlap_cols[-1]

                blend_margin = 50
                blend_start = max(0, blend_start - blend_margin)
                blend_end = min(self.output_size[0], blend_end + blend_margin)

                min_blend_width = 200
                current_width = blend_end - blend_start
                if current_width < min_blend_width:
                    center = (blend_start + blend_end) // 2
                    blend_start = max(0, center - min_blend_width // 2)
                    blend_end = min(self.output_size[0], center + min_blend_width // 2)

        logger.info(f"Область blending'а: {blend_start}-{blend_end} (ширина: {blend_end - blend_start})")

        h, w = self.output_size[1], self.output_size[0]
        self.blend_mask = np.zeros((h, w), dtype=np.float32)

        blend_start = max(0, blend_start)
        blend_end = min(w, blend_end)

        overlap_width = blend_end - blend_start

        if overlap_width > 0:
            for x in range(blend_start, blend_end):
                t = (x - blend_start) / overlap_width
                alpha = 1 / (1 + np.exp(-12 * (t - 0.5)))
                self.blend_mask[:, x] = alpha

        self.blend_mask_3d = np.stack([self.blend_mask] * 3, axis=2)

    def stitch_frame(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Сшивка кадра с предварительно вычисленными трансформациями
        
        Args:
            frame1: Первый кадр
            frame2: Второй кадр
            
        Returns:
            Сшитое изображение
        """
        warped1 = cv2.warpPerspective(frame1, self.final_transform_1, self.output_size,
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped2 = cv2.warpPerspective(frame2, self.final_transform_2, self.output_size,
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        result = np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)

        mask1 = (warped1.sum(axis=2) > 10)
        result[mask1] = warped1[mask1]

        mask2 = (warped2.sum(axis=2) > 10)
        video2_only = mask2 & ~mask1
        result[video2_only] = warped2[video2_only]

        overlap_region = mask1 & mask2
        if np.any(overlap_region):
            blended = warped1 * (1 - self.blend_mask_3d) + warped2 * self.blend_mask_3d
            result[overlap_region] = blended[overlap_region].astype(np.uint8)

        return result

    def create_cylindrical_map(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создание LUT (Look-Up Table) для цилиндрической проекции
        
        Args:
            width: Ширина изображения
            height: Высота изображения
            
        Returns:
            Кортеж (map_x, map_y) для использования в cv2.remap
        """
        f = width / (2 * math.tan(math.radians(self.fov_horizontal / 2)))

        logger.info(f"Создание LUT для цилиндрической проекции:")
        logger.info(f"  Размер: {width}x{height}")
        logger.info(f"  Фокусное расстояние: {f:.1f} пикселей")
        logger.info(f"  Угол обзора: {self.fov_horizontal}°")

        x = np.arange(width, dtype=np.float32)
        y = np.arange(height, dtype=np.float32)

        center_x = width / 2
        center_y = height / 2

        X, Y = np.meshgrid(x, y)

        theta = (X - center_x) / f
        h_cyl = (Y - center_y) / f

        map_x = f * np.tan(theta) + center_x
        map_y = f * h_cyl / np.cos(theta) + center_y

        map_x = np.nan_to_num(map_x, nan=0.0, posinf=width-1, neginf=0.0)
        map_y = np.nan_to_num(map_y, nan=0.0, posinf=height-1, neginf=0.0)

        map_x = np.clip(map_x, 0, width - 1)
        map_y = np.clip(map_y, 0, height - 1)

        return map_x, map_y

    def cylindrical_projection(self, frame: np.ndarray) -> np.ndarray:
        """
        Быстрая цилиндрическая проекция с использованием LUT
        
        Args:
            frame: Входное изображение
            
        Returns:
            Изображение после цилиндрической проекции
            
        Raises:
            ValueError: Если карты проекции не инициализированы
        """
        if self.cylindrical_map_x is None or self.cylindrical_map_y is None:
            logger.error("Карты цилиндрической проекции не инициализированы")
            raise ValueError("Карты цилиндрической проекции не инициализированы")
        
        result = cv2.remap(frame, self.cylindrical_map_x, self.cylindrical_map_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return result

    def find_content_bounds(self, image: np.ndarray, black_threshold: int = 20) -> Tuple[int, int, int, int]:
        """
        Нахождение границ контента в изображении
        
        Args:
            image: Входное изображение
            black_threshold: Порог для определения черных пикселей
            
        Returns:
            Кортеж (left, top, right, bottom) границ контента
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        height, width = gray.shape

        mask = gray > black_threshold

        if not np.any(mask):
            logger.warning("Все пиксели черные, возвращаю полный кадр")
            return 0, 0, width, height

        col_sums = np.sum(mask, axis=0)
        row_sums = np.sum(mask, axis=1)

        left = 0
        for i in range(width):
            if col_sums[i] > 0:
                left = i
                break

        right = width - 1
        for i in range(width - 1, -1, -1):
            if col_sums[i] > 0:
                right = i
                break

        top = 0
        for i in range(height):
            if row_sums[i] > 0:
                top = i
                break

        bottom = height - 1
        for i in range(height - 1, -1, -1):
            if row_sums[i] > 0:
                bottom = i
                break

        padding = 5
        left = max(0, left - padding)
        right = min(width, right + padding)
        top = max(0, top - padding)
        bottom = min(height, bottom + padding)

        if right <= left:
            left, right = 0, width
        if bottom <= top:
            top, bottom = 0, height

        return left, top, right, bottom

    def analyze_and_compute_crop(self, stitched_frame: np.ndarray) -> None:
        """
        Анализ первого кадра для определения параметров обрезки
        
        Args:
            stitched_frame: Сшитый кадр для анализа
        """
        logger.info("Анализ первого кадра для определения обрезки...")
        
        height, width = stitched_frame.shape[:2]
        logger.info(f"Исходный размер сшитого кадра: {width}x{height}")

        cylindrical_frame = self.cylindrical_projection(stitched_frame)

        crop_percent = self.horizontal_crop_percent
        crop_left = int(width * crop_percent)
        crop_right = int(width * crop_percent)

        x1 = crop_left
        y1 = 0
        x2 = width - crop_right
        y2 = height

        x1 = max(0, x1)
        x2 = min(width, x2)
        y1 = max(0, y1)
        y2 = min(height, y2)

        horizontal_cropped = cylindrical_frame[y1:y2, x1:x2]
        self.horizontal_crop_slices = {
            'x_slice': slice(x1, x2),
            'y_slice': slice(y1, y2)
        }

        left, top, right, bottom = self.find_content_bounds(horizontal_cropped, black_threshold=15)

        min_width = 100
        min_height = 100

        if (right - left) < min_width:
            center_x = (left + right) // 2
            left = max(0, center_x - min_width // 2)
            right = min(horizontal_cropped.shape[1], left + min_width)

        if (bottom - top) < min_height:
            center_y = (top + bottom) // 2
            top = max(0, center_y - min_height // 2)
            bottom = min(horizontal_cropped.shape[0], top + min_height)

        self.final_crop_slices = {
            'x_slice': slice(left, right),
            'y_slice': slice(top, bottom)
        }

        final_width = right - left
        final_height = bottom - top

        final_width = final_width if final_width % 2 == 0 else final_width - 1
        final_height = final_height if final_height % 2 == 0 else final_height - 1
        
        final_width = max(min_width, final_width)
        final_height = max(min_height, final_height)
        
        if final_width % 2 != 0:
            final_width += 1
        if final_height % 2 != 0:
            final_height += 1

        self.final_output_size = (final_width, final_height)

        logger.info(f"Финальный размер (четный): {final_width}x{final_height}")

    def apply_crop(self, cylindrical_frame: np.ndarray) -> np.ndarray:
        """
        Применение двухэтапной обрезки к кадру
        
        Args:
            cylindrical_frame: Изображение после цилиндрической проекции
            
        Returns:
            Обрезанное изображение
        """
        if self.horizontal_crop_slices is None or self.final_crop_slices is None:
            logger.warning("Срезы для обрезки не определены, возвращаю исходный кадр")
            return cylindrical_frame

        horizontally_cropped = cylindrical_frame[
            self.horizontal_crop_slices['y_slice'],
            self.horizontal_crop_slices['x_slice']
        ]

        final_cropped = horizontally_cropped[
            self.final_crop_slices['y_slice'],
            self.final_crop_slices['x_slice']
        ]

        return final_cropped

    def create_video_writer(self, output_path: str, fps: float, size: tuple) -> Tuple[cv2.VideoWriter, str]:
        """
        Создание VideoWriter с надежным кодеком
        
        Args:
            output_path: Путь для сохранения видео
            fps: Частота кадров
            size: Размер видео (ширина, высота)
            
        Returns:
            Кортеж (VideoWriter, путь к файлу)
            
        Raises:
            Exception: Если не удалось создать VideoWriter
        """
        width, height = size
        
        if width % 2 != 0:
            width += 1
            logger.info(f"Исправлена ширина: {width}")
        if height % 2 != 0:
            height += 1
            logger.info(f"Исправлена высота: {height}")
            
        size = (width, height)
        logger.info(f"Финальный размер видео: {width}x{height}")

        output_path_with_ext = output_path
        if not output_path.lower().endswith(('.mp4', '.avi')):
            output_path_with_ext = output_path + '.avi'
        
        codec_attempts = [
            ('MJPG', '.avi'), 
            ('XVID', '.avi'), 
            ('mp4v', '.mp4'), 
        ]
        
        for codec, ext in codec_attempts:
            try:
                if not output_path_with_ext.endswith(ext):
                    base_path = os.path.splitext(output_path_with_ext)[0]
                    output_path_with_ext = base_path + ext
                
                logger.info(f"Попытка создания VideoWriter с кодеком {codec}...")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path_with_ext, fourcc, fps, size)
                
                if out.isOpened():
                    logger.info(f"Успешно создан VideoWriter с кодеком {codec}")
                    logger.info(f"Файл: {output_path_with_ext}")
                    return out, output_path_with_ext
                else:
                    out.release()
                    logger.warning(f"Не удалось открыть VideoWriter с кодеком {codec}")
            except Exception as e:
                logger.error(f"Ошибка при создании VideoWriter с кодеком {codec}: {e}")
                continue

        try:
            output_path_with_ext = output_path + '_raw.avi'
            logger.info("Попытка создания VideoWriter с кодеком IYUV (RAW)...")
            fourcc = cv2.VideoWriter_fourcc(*'IYUV')
            out = cv2.VideoWriter(output_path_with_ext, fourcc, fps, size)
            
            if out.isOpened():
                logger.info("Успешно создан VideoWriter с кодеком IYUV (RAW)")
                logger.info(f"Файл: {output_path_with_ext}")
                return out, output_path_with_ext
        except Exception as e:
            logger.error(f"Ошибка при создании RAW VideoWriter: {e}")

        logger.error(f"Не удалось создать VideoWriter. Убедитесь, что размер четный ({width}x{height})")
        raise Exception(f"Не удалось создать VideoWriter. Убедитесь, что размер четный ({width}x{height})")

    def ensure_even_size(self, frame: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        Приведение кадра к целевому размеру
        
        Args:
            frame: Входное изображение
            target_size: Целевой размер (ширина, высота)
            
        Returns:
            Изображение целевого размера
        """
        current_height, current_width = frame.shape[:2]
        target_width, target_height = target_size

        if (current_width == target_width and current_height == target_height):
            return frame

        logger.debug(f"Изменение размера с {current_width}x{current_height} на {target_width}x{target_height}")
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

    def process_full_pipeline(self) -> str:
        """
        Полный пайплайн обработки видео
        
        Returns:
            Путь к созданному видеофайлу или папке с изображениями
            
        Raises:
            Exception: Если возникают критические ошибки
        """
        self.initialize_stitching_parameters()

        cap1 = cv2.VideoCapture(self.video1_path)
        cap2 = cv2.VideoCapture(self.video2_path)

        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        fps = min(fps1, fps2)
        if fps <= 0:
            fps = 30.0
            logger.warning(f"Не удалось определить FPS, использую значение по умолчанию: {fps}")
            
        total_frames = int(min(
            cap1.get(cv2.CAP_PROP_FRAME_COUNT),
            cap2.get(cv2.CAP_PROP_FRAME_COUNT)
        ))

        logger.info(f"Параметры видео:")
        logger.info(f"  Видео 1: {self.video1_path}")
        logger.info(f"  Видео 2: {self.video2_path}")
        logger.info(f"  Частота кадров: {fps:.2f} FPS")
        logger.info(f"  Всего кадров: {total_frames}")
        logger.info(f"  Размер сшивки: {self.output_size[0]}x{self.output_size[1]}")

        logger.info("Анализ первого кадра для определения обрезки...")
        ret1, first_frame1 = cap1.read()
        ret2, first_frame2 = cap2.read()

        if not ret1 or not ret2:
            logger.error("Не удалось прочитать первые кадры")
            raise Exception("Не удалось прочитать первые кадры")

        first_stitched = self.stitch_frame(first_frame1, first_frame2)
        
        logger.info("Создание карт для цилиндрической проекции...")
        self.cylindrical_map_x, self.cylindrical_map_y = self.create_cylindrical_map(
            self.output_size[0], self.output_size[1]
        )
        
        self.analyze_and_compute_crop(first_stitched)

        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

        logger.info("Создание выходного видеофайла...")
        try:
            final_out, final_output_path = self.create_video_writer(
                self.output_path, fps, self.final_output_size
            )
        except Exception as e:
            logger.error(f"Не удалось создать VideoWriter: {e}")
            logger.info("Пробую сохранение как изображений...")
            return self.save_as_images(cap1, cap2, total_frames)

        logger.info(f"Начинаю обработку {total_frames} кадров...")
        logger.info(f"  Финальный размер: {self.final_output_size[0]}x{self.final_output_size[1]}")
        logger.info(f"  Выходной файл: {final_output_path}")

        frame_count = 0
        import time
        start_time = time.time()

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            try:
                stitched = self.stitch_frame(frame1, frame2)
                cylindrical = self.cylindrical_projection(stitched)
                cropped_frame = self.apply_crop(cylindrical)
                final_frame = self.ensure_even_size(cropped_frame, self.final_output_size)
                final_out.write(final_frame)
                frame_count += 1

                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Обработано: {frame_count}/{total_frames} ({progress:.1f}%), "
                              f"Скорость: {fps_actual:.1f} FPS")

            except Exception as e:
                logger.error(f"Ошибка при обработке кадра {frame_count}: {e}")
                continue

        logger.info("Завершение обработки...")
        cap1.release()
        cap2.release()
        final_out.release()

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        logger.info(f"Всего кадров: {frame_count}")
        logger.info(f"Общее время: {total_time:.1f} секунд")
        logger.info(f"Средняя скорость: {avg_fps:.1f} FPS")
        logger.info(f"Финальный размер: {self.final_output_size[0]}x{self.final_output_size[1]}")
        logger.info(f"Финальный файл: {final_output_path}")

        if os.path.exists(final_output_path):
            file_size = os.path.getsize(final_output_path) / (1024 * 1024)
            logger.info(f"Размер файла: {file_size:.2f} MB")
            
            test_cap = cv2.VideoCapture(final_output_path)
            if test_cap.isOpened():
                test_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                test_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                test_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                test_cap.release()
                logger.info("Файл успешно открыт OpenCV")
                logger.info(f"  Кадров: {test_frames}, Размер: {test_width}x{test_height}")
            else:
                logger.warning("Файл создан, но OpenCV не может его открыть")
                logger.info("Попробуйте открыть в медиаплеере (VLC, Windows Media Player)")

        return final_output_path

    def initialize_stitching_parameters(self) -> None:
        """
        Инициализация всех параметров сшивки
        """
        logger.info("Инициализация параметров сшивки...")
        
        self.calculate_homography_from_frames()

        cap1 = cv2.VideoCapture(self.video1_path)
        cap2 = cv2.VideoCapture(self.video2_path)

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            logger.error("Не удалось прочитать кадры для инициализации")
            raise Exception("Не удалось прочитать кадры для инициализации")

        self.neutral_transform_2 = self.neutral_plane_transform()
        self.calculate_new_homography_to_neutral_plane(frame1, frame2)
        self.calculate_final_transforms(frame1, frame2)
        self.precompute_blend_mask(frame1, frame2)

        cap1.release()
        cap2.release()

        logger.info("Параметры сшивки инициализированы!")

    def save_as_images(self, cap1, cap2, total_frames):
        """
        Сохранение кадров как изображений (fallback метод)
        
        Args:
            cap1: VideoCapture для первого видео
            cap2: VideoCapture для второго видео
            total_frames: Общее количество кадров
            
        Returns:
            Путь к папке с сохраненными изображениями
        """
        logger.info("Использую fallback: сохранение кадров как изображений...")

        output_dir = "stitched_frames"
        os.makedirs(output_dir, exist_ok=True)

        frame_count = 0
        import time
        start_time = time.time()

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            try:
                stitched = self.stitch_frame(frame1, frame2)
                cylindrical = self.cylindrical_projection(stitched)
                cropped_frame = self.apply_crop(cylindrical)
                final_frame = self.ensure_even_size(cropped_frame, self.final_output_size)

                frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, final_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

                frame_count += 1

                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Сохранено: {frame_count}/{total_frames} ({progress:.1f}%), "
                              f"Скорость: {fps_actual:.1f} FPS")

            except Exception as e:
                logger.error(f"Ошибка при сохранении кадра {frame_count}: {e}")
                continue

        cap1.release()
        cap2.release()

        logger.info(f"Сохранено {frame_count} кадров в папку '{output_dir}'")
        logger.info(f"Финальный размер: {self.final_output_size[0]}x{self.final_output_size[1]}")

        return output_dir


def main():
    """
    Основная функция для запуска обработки видео
    
    Пример использования:
        processor = OptimizedCylindricalStitcher(...)
        result = processor.process_full_pipeline()
    """
    video1_path = "v1_change2.8.mp4"
    video2_path = "v2_dist.mp4"
    output_path = "final_cylindrical_panorama"

    logger.info(f"Запуск обработки видео:")
    logger.info(f"  Видео 1: {video1_path}")
    logger.info(f"  Видео 2: {video2_path}")
    logger.info(f"  Выходной файл: {output_path}")

    try:
        processor = OptimizedCylindricalStitcher(
            video1_path=video1_path,
            video2_path=video2_path,
            output_path=output_path,
            num_calibration_frames=15,
            neutral_plane_t=0.493,
            fov_horizontal=150
        )

        final_video_path = processor.process_full_pipeline()

        if final_video_path:
            logger.info("=" * 60)
            logger.info("ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
            logger.info("=" * 60)

            if isinstance(final_video_path, str) and final_video_path.endswith(('.mp4', '.avi')):
                logger.info(f"Видео сохранено: {final_video_path}")
                logger.info(f"Размер видео: {processor.final_output_size[0]}x{processor.final_output_size[1]}")
                logger.info("Проверьте файл в медиаплеере (VLC, Windows Media Player)")
            else:
                logger.info(f"Кадры сохранены в папку: {final_video_path}")
                logger.info(f"Размер кадров: {processor.final_output_size[0]}x{processor.final_output_size[1]}")
        else:
            logger.error("Ошибка при обработке видео")
            
    except Exception as e:
        logger.error(f"Критическая ошибка при обработке: {e}")
        raise


if __name__ == "__main__":
    main()