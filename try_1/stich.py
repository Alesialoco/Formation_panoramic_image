import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import os
import math
import logging
import sys
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


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
                 fov_horizontal: float = 150, adaptive_smoothness: float = 50.0, 
                 crop_percent: float = 0.15):
        """
        Инициализация класса для сшивки видео
        
        Args:
            video1_path: Путь к первому видео
            video2_path: Путь ко второму видео
            output_path: Путь для сохранения результата
            num_calibration_frames: Количество кадров для калибровки гомографии
            neutral_plane_t: Параметр нейтральной плоскости (0-1)
            fov_horizontal: Горизонтальный угол обзора для проекции
            adaptive_smoothness: Параметр гладкости для адаптивного масштабирования (1-100)
            crop_percent: Процент обрезки с каждой стороны (0.0-0.5)
            
        Raises:
            ValueError: Если neutral_plane_t не в диапазоне [0, 1]
        """
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.output_path = output_path
        self.num_calibration_frames = num_calibration_frames
        self.neutral_plane_t = neutral_plane_t
        self.fov_horizontal = fov_horizontal
        self.adaptive_smoothness = adaptive_smoothness
        self.crop_percent = max(0.0, min(0.5, crop_percent))  # Ограничиваем от 0 до 0.5

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

        # Параметры проекции и адаптивного масштабирования
        self.projection_map_x = None
        self.projection_map_y = None
        self.adaptive_params = None
        self.final_output_size = None
        
        # Параметры для адаптивного масштабирования
        self.top_boundary_smooth = None
        self.bottom_boundary_smooth = None
        self.target_height = None
        self.adaptive_map_x = None
        self.adaptive_map_y = None
        
        # Параметры для обрезки
        self.crop_left = 0
        self.crop_right = 0
        self.crop_top = 0
        self.crop_bottom = 0
        self.cropped_size = None
        
        # Минимальная высота после адаптивного масштабирования
        self.min_height = 150
        
        # Процент центральной области для анализа границ (0.8 = 80%)
        self.center_analysis_percent = 0.8

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Извлечение признаков SIFT из изображения
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def find_homography(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        """
        Поиск матрицы гомографии между двумя изображениями
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
        """
        logger.info(f"Создание преобразования для нейтральной плоскости (t={self.neutral_plane_t})...")

        H2_to_1 = self.homography_matrix_2_to_1 / self.homography_matrix_2_to_1[2, 2]

        H2_neutral = (np.eye(3) * (1 - self.neutral_plane_t) + H2_to_1 * self.neutral_plane_t)
        H2_neutral = H2_neutral / H2_neutral[2, 2]

        return H2_neutral

    def calculate_new_homography_to_neutral_plane(self, frame1: np.ndarray, frame2: np.ndarray) -> None:
        """
        Вычисление новой гомографии к нейтральной плоскости
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

    def apply_projection(self, frame: np.ndarray) -> np.ndarray:
        """
        Применение проекции (цилиндрической или сферической) с использованием LUT
        """
        if self.projection_map_x is None or self.projection_map_y is None:
            logger.error("Карты проекции не инициализированы")
            raise ValueError("Карты проекции не инициализированы")
        
        result = cv2.remap(frame, self.projection_map_x, self.projection_map_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return result

    def remove_black_borders(self, image: np.ndarray, threshold: int = 10) -> Tuple[int, int, int, int]:
        """
        Удаление черных границ вокруг изображения
        
        Args:
            image: Входное изображение
            threshold: Порог для определения черных пикселей
            
        Returns:
            Кортеж (left, top, right, bottom) - координаты для обрезки
        """
        h, w = image.shape[:2]
        
        # Конвертируем в градации серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Создаем маску нечерных пикселей
        mask = gray > threshold
        
        if not np.any(mask):
            logger.warning("Изображение полностью черное, возвращаю исходные границы")
            return 0, 0, w, h
        
        # Находим границы по вертикали и горизонтали
        col_sums = np.sum(mask, axis=0)
        row_sums = np.sum(mask, axis=1)
        
        # Левая граница
        left = 0
        for i in range(w):
            if col_sums[i] > 0:
                left = i
                break
        
        # Правая граница
        right = w - 1
        for i in range(w - 1, -1, -1):
            if col_sums[i] > 0:
                right = i
                break
        
        # Верхняя граница
        top = 0
        for i in range(h):
            if row_sums[i] > 0:
                top = i
                break
        
        # Нижняя граница
        bottom = h - 1
        for i in range(h - 1, -1, -1):
            if row_sums[i] > 0:
                bottom = i
                break
        
        # Добавляем небольшой отступ для безопасности
        margin = 2
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(w, right + margin)
        bottom = min(h, bottom + margin)
        
        logger.debug(f"Границы контента: left={left}, top={top}, right={right}, bottom={bottom}")
        
        return left, top, right, bottom

    def apply_side_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Применение обрезки боковых артефактов по заданному проценту
        
        Args:
            image: Входное изображение
            
        Returns:
            Изображение с обрезанными боками
        """
        h, w = image.shape[:2]
        
        # Вычисляем количество пикселей для обрезки с каждой стороны
        crop_pixels = int(w * self.crop_percent)
        
        # Применяем обрезку
        if crop_pixels > 0 and w > 2 * crop_pixels:
            cropped = image[:, crop_pixels:w - crop_pixels]
            logger.debug(f"Обрезка боков: {crop_pixels} пикселей с каждой стороны")
            return cropped
        else:
            logger.warning(f"Слишком большой процент обрезки ({self.crop_percent}), пропускаю")
            return image

    def ensure_even_size_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Обрезка изображения до четных размеров
        
        Args:
            image: Входное изображение
            
        Returns:
            Изображение с четными размерами
        """
        h, w = image.shape[:2]
        
        new_w = w if w % 2 == 0 else w - 1
        new_h = h if h % 2 == 0 else h - 1
        
        if new_w != w or new_h != h:
            # Обрезаем до четных размеров
            image = image[:new_h, :new_w]
            logger.debug(f"Коррекция до четных размеров: {new_w}x{new_h}")
        
        return image

    def analyze_panorama_boundaries(self, panorama: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Анализ верхней и нижней границ панорамы для всех столбцов
        
        Args:
            panorama: Входная панорама
            
        Returns:
            Кортеж (top_boundary, bottom_boundary) - границы для каждого столбца
        """
        h, w = panorama.shape[:2]
        
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        top_boundary = np.zeros(w)
        bottom_boundary = np.zeros(w)
        
        for col in range(w):
            col_data = binary[:, col]
            non_zero = np.where(col_data > 0)[0]
            
            if len(non_zero) > 0:
                top_boundary[col] = non_zero[0]
                bottom_boundary[col] = non_zero[-1]
            else:
                top_boundary[col] = 0
                bottom_boundary[col] = h - 1
        
        return top_boundary, bottom_boundary

    def compute_target_height_from_center(self, top_boundary: np.ndarray, bottom_boundary: np.ndarray, w: int) -> int:
        """
        Вычисление целевой высоты только по центральной области
        
        Args:
            top_boundary: Массив верхних границ для всех столбцов
            bottom_boundary: Массив нижних границ для всех столбцов
            w: Ширина изображения
            
        Returns:
            Целевая высота
        """
        # Определяем центральную область
        center_start = int(w * (1 - self.center_analysis_percent) / 2)
        center_end = int(w * (1 + self.center_analysis_percent) / 2)
        
        # Берем границы только из центральной области
        center_top = top_boundary[center_start:center_end]
        center_bottom = bottom_boundary[center_start:center_end]
        
        # Вычисляем целевую высоту по центральной области
        target_top = np.max(center_top)
        target_bottom = np.min(center_bottom)
        target_height = int(target_bottom - target_top)
        
        logger.info(f"Вычисление целевой высоты по центральной области:")
        logger.info(f"  Столбцы {center_start}-{center_end} из {w} (центральные {self.center_analysis_percent*100:.0f}%)")
        logger.info(f"  Максимальная верхняя граница в центре: {target_top:.0f}")
        logger.info(f"  Минимальная нижняя граница в центре: {target_bottom:.0f}")
        logger.info(f"  Целевая высота: {target_height} пикселей")
        
        return target_height

    def smooth_boundaries(self, top_boundary: np.ndarray, bottom_boundary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сглаживание границ с заданным уровнем гладкости
        """
        h = len(top_boundary)
        sigma = self.adaptive_smoothness / 5.0
        
        top_smooth = gaussian_filter1d(top_boundary, sigma=sigma, mode='reflect')
        bottom_smooth = gaussian_filter1d(bottom_boundary, sigma=sigma, mode='reflect')
        
        top_smooth = np.clip(top_smooth, 0, h-1)
        bottom_smooth = np.clip(bottom_smooth, 0, h-1)
        
        for i in range(len(top_smooth)):
            if top_smooth[i] >= bottom_smooth[i]:
                mid = (top_boundary[i] + bottom_boundary[i]) / 2
                top_smooth[i] = mid - 10
                bottom_smooth[i] = mid + 10
        
        return top_smooth, bottom_smooth

    def create_adaptive_height_map(self, panorama: np.ndarray, top_smooth: np.ndarray, 
                                   bottom_smooth: np.ndarray, target_height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создание карты трансформации для адаптивного масштабирования по высоте
        
        Args:
            panorama: Исходная панорама
            top_smooth: Сглаженные верхние границы
            bottom_smooth: Сглаженные нижние границы
            target_height: Целевая высота (уже вычислена)
            
        Returns:
            Кортеж (map_x, map_y) - карты трансформации
        """
        h, w = panorama.shape[:2]
        
        # Создаем карты трансформации для всего изображения
        map_x = np.zeros((target_height, w), dtype=np.float32)
        map_y = np.zeros((target_height, w), dtype=np.float32)
        
        for col in range(w):
            current_top = top_smooth[col]
            current_bottom = bottom_smooth[col]
            current_height = current_bottom - current_top
            
            if current_height <= 0:
                current_height = target_height
            
            scale = target_height / current_height
            
            for row in range(target_height):
                src_y = current_top + (row / scale)
                src_y = np.clip(src_y, 0, h-1)
                
                map_x[row, col] = col
                map_y[row, col] = src_y
        
        return map_x, map_y

    def apply_adaptive_scaling(self, panorama: np.ndarray) -> np.ndarray:
        """
        Применение адаптивного масштабирования к панораме
        """
        h, w = panorama.shape[:2]
        
        # Анализируем границы для всех столбцов
        top, bottom = self.analyze_panorama_boundaries(panorama)
        
        # Вычисляем целевую высоту ТОЛЬКО по центральной области
        target_height = self.compute_target_height_from_center(top, bottom, w)
        
        # Применяем минимальную отметку высоты
        original_target_height = target_height
        if target_height < self.min_height:
            target_height = self.min_height
            logger.warning(f"Высота после адаптивного масштабирования была {original_target_height}px, "
                          f"увеличена до минимальной {self.min_height}px")
        
        # Убеждаемся, что высота четная
        if target_height % 2 != 0:
            target_height += 1
            logger.debug(f"Высота скорректирована до четной: {target_height}")
        
        logger.info(f"Итоговая целевая высота: {target_height} пикселей")
        
        # Сглаживаем границы для плавного перехода
        top_smooth, bottom_smooth = self.smooth_boundaries(top, bottom)
        
        self.top_boundary_smooth = top_smooth
        self.bottom_boundary_smooth = bottom_smooth
        self.target_height = target_height
        
        # Создаем карты трансформации с единой целевой высотой для всех столбцов
        map_x, map_y = self.create_adaptive_height_map(
            panorama, top_smooth, bottom_smooth, target_height
        )
        
        self.adaptive_map_x = map_x
        self.adaptive_map_y = map_y
        
        result = cv2.remap(
            panorama, 
            map_x.astype(np.float32), 
            map_y.astype(np.float32), 
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return result

    def analyze_and_compute_crop_params(self, stitched_frame: np.ndarray) -> None:
        """
        Анализ первого кадра для определения всех параметров обрезки и масштабирования
        
        Args:
            stitched_frame: Сшитый кадр для анализа
        """
        logger.info("Анализ первого кадра для определения параметров обработки...")
        logger.info(f"Процент обрезки боков: {self.crop_percent * 100:.1f}%")
        logger.info(f"Минимальная высота после масштабирования: {self.min_height}px")
        logger.info(f"Центральная область для определения высоты: {self.center_analysis_percent * 100:.0f}%")
        
        # Шаг 1: Применяем проекцию
        projected_frame = self.apply_projection(stitched_frame)
        logger.info(f"После проекции: {projected_frame.shape[1]}x{projected_frame.shape[0]}")
        
        # Шаг 2: Применяем обрезку боковых артефактов
        side_cropped = self.apply_side_crop(projected_frame)
        logger.info(f"После обрезки боков: {side_cropped.shape[1]}x{side_cropped.shape[0]}")
        
        # Шаг 3: Удаляем черные границы
        left, top, right, bottom = self.remove_black_borders(side_cropped)
        self.crop_left, self.crop_top, self.crop_right, self.crop_bottom = left, top, right, bottom
        
        borders_removed = side_cropped[top:bottom, left:right]
        logger.info(f"После удаления черных границ: {borders_removed.shape[1]}x{borders_removed.shape[0]}")
        
        # Шаг 4: Применяем адаптивное масштабирование по высоте
        scaled = self.apply_adaptive_scaling(borders_removed)
        logger.info(f"После адаптивного масштабирования: {scaled.shape[1]}x{scaled.shape[0]}")

        # Шаг 5: Проверяем четность размеров
        final = self.ensure_even_size_crop(scaled)
        
        self.final_output_size = (final.shape[1], final.shape[0])
        
        logger.info(f"Финальный размер после всех этапов: {self.final_output_size[0]}x{self.final_output_size[1]}")

    def process_frame_full_pipeline(self, frame: np.ndarray) -> np.ndarray:
        """
        Полный пайплайн обработки одного кадра
        
        Args:
            frame: Входной кадр
            
        Returns:
            Обработанный кадр
        """
        # Шаг 1: Проекция
        projected = self.apply_projection(frame)
        
        # Шаг 2: Обрезка боковых артефактов
        side_cropped = self.apply_side_crop(projected)
        
        # Шаг 3: Удаление черных границ (используем предвычисленные координаты)
        borders_removed = side_cropped[
            self.crop_top:self.crop_bottom, 
            self.crop_left:self.crop_right
        ]
        
        # Шаг 4: Адаптивное масштабирование (используем предвычисленные карты)
        if self.adaptive_map_x is not None and self.adaptive_map_y is not None:
            scaled = cv2.remap(
                borders_removed,
                self.adaptive_map_x.astype(np.float32),
                self.adaptive_map_y.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            scaled = self.apply_adaptive_scaling(borders_removed)

        # Шаг 5: Проверка четности размеров
        final = self.ensure_even_size_crop(scaled)
        
        return final

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

    def process_full_pipeline(self) -> str:
        """
        Полный пайплайн обработки видео
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
        logger.info(f"  Параметр гладкости: {self.adaptive_smoothness}")
        logger.info(f"  Процент обрезки боков: {self.crop_percent * 100:.1f}%")

        logger.info("Анализ первого кадра для определения параметров обработки...")
        ret1, first_frame1 = cap1.read()
        ret2, first_frame2 = cap2.read()

        if not ret1 or not ret2:
            logger.error("Не удалось прочитать первые кадры")
            raise Exception("Не удалось прочитать первые кадры")

        first_stitched = self.stitch_frame(first_frame1, first_frame2)
        
        logger.info("Создание карт для проекции...")
        self.projection_map_x, self.projection_map_y = self.create_cylindrical_map(
            self.output_size[0], self.output_size[1])
        
        # Анализируем и вычисляем все параметры обработки
        self.analyze_and_compute_crop_params(first_stitched)

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
                processed_frame = self.process_frame_full_pipeline(stitched)
                
                # Финальная проверка размера
                if (processed_frame.shape[1], processed_frame.shape[0]) != self.final_output_size:
                    processed_frame = cv2.resize(processed_frame, self.final_output_size)
                
                final_out.write(processed_frame)
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

        return final_output_path

    def create_video_writer(self, output_path: str, fps: float, size: tuple) -> Tuple[cv2.VideoWriter, str]:
        """
        Создание VideoWriter с надежным кодеком
        """
        width, height = size
        
        # Убеждаемся, что размеры четные
        if width % 2 != 0:
            width -= 1
            logger.info(f"Исправлена ширина: {width}")
        if height % 2 != 0:
            height -= 1
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
        """
        current_height, current_width = frame.shape[:2]
        target_width, target_height = target_size

        if (current_width == target_width and current_height == target_height):
            return frame

        logger.debug(f"Изменение размера с {current_width}x{current_height} на {target_width}x{target_height}")
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

    def save_as_images(self, cap1, cap2, total_frames):
        """
        Сохранение кадров как изображений (fallback метод)
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
                processed_frame = self.process_frame_full_pipeline(stitched)
                
                # Финальная проверка размера
                if (processed_frame.shape[1], processed_frame.shape[0]) != self.final_output_size:
                    processed_frame = cv2.resize(processed_frame, self.final_output_size)

                frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

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