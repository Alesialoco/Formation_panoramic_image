import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import os
import math
import logging
import sys
import pickle
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from dataclasses import dataclass, asdict
import json


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class StitchingParameters2:
    """Класс для хранения всех параметров сшивки"""
    # Матрицы трансформаций
    homography_matrix_1_to_2: np.ndarray
    homography_matrix_2_to_1: np.ndarray
    neutral_transform_2: np.ndarray
    neutral_transform_2_corrected: np.ndarray
    new_homography_1_to_2_neutral: np.ndarray
    final_transform_1: np.ndarray
    final_transform_2: np.ndarray
    
    # Параметры вывода
    output_size: Tuple[int, int]
    blend_mask: np.ndarray
    blend_mask_3d: np.ndarray
    
    # Параметры проекции
    projection_map_x: np.ndarray
    projection_map_y: np.ndarray
    
    # Параметры адаптивного масштабирования
    adaptive_map_x: np.ndarray
    adaptive_map_y: np.ndarray
    top_boundary_smooth: np.ndarray
    bottom_boundary_smooth: np.ndarray
    target_height: int
    
    # Параметры обрезки
    crop_left: int
    crop_top: int
    crop_right: int
    crop_bottom: int
    final_output_size: Tuple[int, int]
    
    # Параметры сшивки
    fov_horizontal: float
    adaptive_smoothness: float
    crop_percent: float
    min_height: int
    center_analysis_percent: float
    
    def save(self, filepath: str):
        """Сохранить параметры в файл"""
        # Преобразуем numpy массивы в сериализуемый формат
        data = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                data[key] = {
                    'type': 'numpy',
                    'dtype': str(value.dtype),
                    'shape': value.shape,
                    'data': value.tobytes()
                }
            elif isinstance(value, tuple):
                data[key] = {
                    'type': 'tuple',
                    'data': list(value)
                }
            else:
                data[key] = {
                    'type': 'native',
                    'data': value
                }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Параметры сшивки сохранены в {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'StitchingParameters':
        """Загрузить параметры из файла"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Восстанавливаем объекты из сериализованного формата
        kwargs = {}
        for key, value in data.items():
            if value['type'] == 'numpy':
                kwargs[key] = np.frombuffer(value['data'], dtype=np.dtype(value['dtype'])).reshape(value['shape'])
            elif value['type'] == 'tuple':
                kwargs[key] = tuple(value['data'])
            else:
                kwargs[key] = value['data']
        
        return cls(**kwargs)


class OptimizedCylindricalStitcher2:
    """
    Класс для сшивки двух видео с цилиндрической проекцией
    """
    
    def __init__(self, video1_path: str = None, video2_path: str = None, output_path: str = None,
                 num_calibration_frames: int = 10, neutral_plane_t: float = 0.5,
                 fov_horizontal: float = 150, adaptive_smoothness: float = 50.0, 
                 crop_percent: float = 0.15):
        """
        Инициализация класса для сшивки видео
        """
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.output_path = output_path
        self.num_calibration_frames = num_calibration_frames
        self.neutral_plane_t = neutral_plane_t
        self.fov_horizontal = fov_horizontal
        self.adaptive_smoothness = adaptive_smoothness
        self.crop_percent = max(0.0, min(0.5, crop_percent))

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
        self.crop_top = 0
        self.crop_right = 0
        self.crop_bottom = 0
        self.cropped_size = None
        
        # Минимальная высота после адаптивного масштабирования
        self.min_height = 150
        
        # Процент центральной области для анализа границ (0.8 = 80%)
        self.center_analysis_percent = 0.8

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Извлечение признаков SIFT из изображения"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def find_homography(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        """Поиск матрицы гомографии между двумя изображениями"""
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
        """Вычисление матриц гомографии на основе нескольких кадров из видео"""
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
        """Создание преобразования для нейтральной плоскости"""
        logger.info(f"Создание преобразования для нейтральной плоскости (t={self.neutral_plane_t})...")

        H2_to_1 = self.homography_matrix_2_to_1 / self.homography_matrix_2_to_1[2, 2]

        H2_neutral = (np.eye(3) * (1 - self.neutral_plane_t) + H2_to_1 * self.neutral_plane_t)
        H2_neutral = H2_neutral / H2_neutral[2, 2]

        return H2_neutral

    def calculate_new_homography_to_neutral_plane(self, frame1: np.ndarray, frame2: np.ndarray) -> None:
        """Вычисление новой гомографии к нейтральной плоскости"""
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
        """Вычисление финальных трансформаций для сшивки"""
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
        """Предварительное вычисление маски для плавного смешивания"""
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
        """Сшивка кадра с предварительно вычисленными трансформациями"""
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
        """Создание LUT (Look-Up Table) для цилиндрической проекции"""
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
        """Применение проекции (цилиндрической или сферической) с использованием LUT"""
        if self.projection_map_x is None or self.projection_map_y is None:
            logger.error("Карты проекции не инициализированы")
            raise ValueError("Карты проекции не инициализированы")
        
        result = cv2.remap(frame, self.projection_map_x, self.projection_map_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return result

    def remove_black_borders(self, image: np.ndarray, threshold: int = 10) -> Tuple[int, int, int, int]:
        """Удаление черных границ вокруг изображения"""
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        mask = gray > threshold
        
        if not np.any(mask):
            logger.warning("Изображение полностью черное, возвращаю исходные границы")
            return 0, 0, w, h
        
        col_sums = np.sum(mask, axis=0)
        row_sums = np.sum(mask, axis=1)
        
        left = 0
        for i in range(w):
            if col_sums[i] > 0:
                left = i
                break
        
        right = w - 1
        for i in range(w - 1, -1, -1):
            if col_sums[i] > 0:
                right = i
                break
        
        top = 0
        for i in range(h):
            if row_sums[i] > 0:
                top = i
                break
        
        bottom = h - 1
        for i in range(h - 1, -1, -1):
            if row_sums[i] > 0:
                bottom = i
                break
        
        margin = 2
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(w, right + margin)
        bottom = min(h, bottom + margin)
        
        logger.debug(f"Границы контента: left={left}, top={top}, right={right}, bottom={bottom}")
        
        return left, top, right, bottom

    def apply_side_crop(self, image: np.ndarray) -> np.ndarray:
        """Применение обрезки боковых артефактов по заданному проценту"""
        h, w = image.shape[:2]
        
        crop_pixels = int(w * self.crop_percent)
        
        if crop_pixels > 0 and w > 2 * crop_pixels:
            cropped = image[:, crop_pixels:w - crop_pixels]
            logger.debug(f"Обрезка боков: {crop_pixels} пикселей с каждой стороны")
            return cropped
        else:
            logger.warning(f"Слишком большой процент обрезки ({self.crop_percent}), пропускаю")
            return image

    def ensure_even_size_crop(self, image: np.ndarray) -> np.ndarray:
        """Обрезка изображения до четных размеров"""
        h, w = image.shape[:2]
        
        new_w = w if w % 2 == 0 else w - 1
        new_h = h if h % 2 == 0 else h - 1
        
        if new_w != w or new_h != h:
            image = image[:new_h, :new_w]
            logger.debug(f"Коррекция до четных размеров: {new_w}x{new_h}")
        
        return image

    def analyze_panorama_boundaries(self, panorama: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Анализ верхней и нижней границ панорамы для всех столбцов"""
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
        """Вычисление целевой высоты только по центральной области"""
        center_start = int(w * (1 - self.center_analysis_percent) / 2)
        center_end = int(w * (1 + self.center_analysis_percent) / 2)
        
        center_top = top_boundary[center_start:center_end]
        center_bottom = bottom_boundary[center_start:center_end]
        
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
        """Сглаживание границ с заданным уровнем гладкости"""
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
        """Создание карты трансформации для адаптивного масштабирования по высоте"""
        h, w = panorama.shape[:2]
        
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
        """Применение адаптивного масштабирования к панораме"""
        h, w = panorama.shape[:2]
        
        top, bottom = self.analyze_panorama_boundaries(panorama)
        
        target_height = self.compute_target_height_from_center(top, bottom, w)
        
        original_target_height = target_height
        if target_height < self.min_height:
            target_height = self.min_height
            logger.warning(f"Высота после адаптивного масштабирования была {original_target_height}px, "
                          f"увеличена до минимальной {self.min_height}px")
        
        if target_height % 2 != 0:
            target_height += 1
            logger.debug(f"Высота скорректирована до четной: {target_height}")
        
        logger.info(f"Итоговая целевая высота: {target_height} пикселей")
        
        top_smooth, bottom_smooth = self.smooth_boundaries(top, bottom)
        
        self.top_boundary_smooth = top_smooth
        self.bottom_boundary_smooth = bottom_smooth
        self.target_height = target_height
        
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
        """Анализ первого кадра для определения всех параметров обрезки и масштабирования"""
        logger.info("Анализ первого кадра для определения параметров обработки...")
        logger.info(f"Процент обрезки боков: {self.crop_percent * 100:.1f}%")
        logger.info(f"Минимальная высота после масштабирования: {self.min_height}px")
        logger.info(f"Центральная область для определения высоты: {self.center_analysis_percent * 100:.0f}%")
        
        projected_frame = self.apply_projection(stitched_frame)
        logger.info(f"После проекции: {projected_frame.shape[1]}x{projected_frame.shape[0]}")
        
        side_cropped = self.apply_side_crop(projected_frame)
        logger.info(f"После обрезки боков: {side_cropped.shape[1]}x{side_cropped.shape[0]}")
        
        left, top, right, bottom = self.remove_black_borders(side_cropped)
        self.crop_left, self.crop_top, self.crop_right, self.crop_bottom = left, top, right, bottom
        
        borders_removed = side_cropped[top:bottom, left:right]
        logger.info(f"После удаления черных границ: {borders_removed.shape[1]}x{borders_removed.shape[0]}")
        
        scaled = self.apply_adaptive_scaling(borders_removed)
        logger.info(f"После адаптивного масштабирования: {scaled.shape[1]}x{scaled.shape[0]}")

        final = self.ensure_even_size_crop(scaled)
        
        self.final_output_size = (final.shape[1], final.shape[0])
        
        logger.info(f"Финальный размер после всех этапов: {self.final_output_size[0]}x{self.final_output_size[1]}")

    def process_frame_full_pipeline(self, frame: np.ndarray) -> np.ndarray:
        """Полный пайплайн обработки одного кадра"""
        projected = self.apply_projection(frame)
        
        side_cropped = self.apply_side_crop(projected)
        
        borders_removed = side_cropped[
            self.crop_top:self.crop_bottom, 
            self.crop_left:self.crop_right
        ]
        
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

        final = self.ensure_even_size_crop(scaled)
        
        return final

    def initialize_stitching_parameters(self) -> None:
        """Инициализация всех параметров сшивки"""
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

    def get_parameters(self) -> StitchingParameters2:
        """Получить все параметры для сохранения"""
        return StitchingParameters2(
            homography_matrix_1_to_2=self.homography_matrix_1_to_2,
            homography_matrix_2_to_1=self.homography_matrix_2_to_1,
            neutral_transform_2=self.neutral_transform_2,
            neutral_transform_2_corrected=self.neutral_transform_2_corrected,
            new_homography_1_to_2_neutral=self.new_homography_1_to_2_neutral,
            final_transform_1=self.final_transform_1,
            final_transform_2=self.final_transform_2,
            output_size=self.output_size,
            blend_mask=self.blend_mask,
            blend_mask_3d=self.blend_mask_3d,
            projection_map_x=self.projection_map_x,
            projection_map_y=self.projection_map_y,
            adaptive_map_x=self.adaptive_map_x,
            adaptive_map_y=self.adaptive_map_y,
            top_boundary_smooth=self.top_boundary_smooth,
            bottom_boundary_smooth=self.bottom_boundary_smooth,
            target_height=self.target_height,
            crop_left=self.crop_left,
            crop_top=self.crop_top,
            crop_right=self.crop_right,
            crop_bottom=self.crop_bottom,
            final_output_size=self.final_output_size,
            fov_horizontal=self.fov_horizontal,
            adaptive_smoothness=self.adaptive_smoothness,
            crop_percent=self.crop_percent,
            min_height=self.min_height,
            center_analysis_percent=self.center_analysis_percent
        )

    def set_parameters(self, params: StitchingParameters2):
        """Установить параметры из загруженного объекта"""
        self.homography_matrix_1_to_2 = params.homography_matrix_1_to_2
        self.homography_matrix_2_to_1 = params.homography_matrix_2_to_1
        self.neutral_transform_2 = params.neutral_transform_2
        self.neutral_transform_2_corrected = params.neutral_transform_2_corrected
        self.new_homography_1_to_2_neutral = params.new_homography_1_to_2_neutral
        self.final_transform_1 = params.final_transform_1
        self.final_transform_2 = params.final_transform_2
        self.output_size = params.output_size
        self.blend_mask = params.blend_mask
        self.blend_mask_3d = params.blend_mask_3d
        self.projection_map_x = params.projection_map_x
        self.projection_map_y = params.projection_map_y
        self.adaptive_map_x = params.adaptive_map_x
        self.adaptive_map_y = params.adaptive_map_y
        self.top_boundary_smooth = params.top_boundary_smooth
        self.bottom_boundary_smooth = params.bottom_boundary_smooth
        self.target_height = params.target_height
        self.crop_left = params.crop_left
        self.crop_top = params.crop_top
        self.crop_right = params.crop_right
        self.crop_bottom = params.crop_bottom
        self.final_output_size = params.final_output_size
        self.fov_horizontal = params.fov_horizontal
        self.adaptive_smoothness = params.adaptive_smoothness
        self.crop_percent = params.crop_percent
        self.min_height = params.min_height
        self.center_analysis_percent = params.center_analysis_percent

    def calibrate(self, calibration_file: str) -> None:
        """Выполнить калибровку и сохранить параметры"""
        self.initialize_stitching_parameters()
        
        cap1 = cv2.VideoCapture(self.video1_path)
        cap2 = cv2.VideoCapture(self.video2_path)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            logger.error("Не удалось прочитать кадры для калибровки")
            raise Exception("Не удалось прочитать кадры для калибровки")
        
        first_stitched = self.stitch_frame(frame1, frame2)
        
        self.projection_map_x, self.projection_map_y = self.create_cylindrical_map(
            self.output_size[0], self.output_size[1])
        
        self.analyze_and_compute_crop_params(first_stitched)
        
        params = self.get_parameters()
        params.save(calibration_file)
        
        cap1.release()
        cap2.release()
        
        logger.info(f"Калибровка завершена. Параметры сохранены в {calibration_file}")

    def process_with_params(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Обработка пары кадров с использованием загруженных параметров"""
        stitched = self.stitch_frame(frame1, frame2)
        processed = self.process_frame_full_pipeline(stitched)

        if (processed.shape[1], processed.shape[0]) != self.final_output_size:
            processed = cv2.resize(processed, self.final_output_size)
        
        return processed
