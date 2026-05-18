import cv2
import numpy as np
from typing import Tuple, List, Optional
import os
import math
import logging
import sys
import pickle
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class StitchingParameters3:
    """Класс для хранения всех параметров сшивки"""
    homography_left_to_center: np.ndarray
    homography_right_to_center: np.ndarray
    final_transform_left: np.ndarray
    final_transform_center: np.ndarray
    final_transform_right: np.ndarray
    output_size: Tuple[int, int]
    blend_mask_left: np.ndarray
    blend_mask_right: np.ndarray
    projection_map_x: np.ndarray
    projection_map_y: np.ndarray
    adaptive_map_x: np.ndarray
    adaptive_map_y: np.ndarray
    top_boundary_smooth: np.ndarray
    bottom_boundary_smooth: np.ndarray
    target_height: int
    crop_left: int
    crop_top: int
    crop_right: int
    crop_bottom: int
    final_output_size: Tuple[int, int]
    fov_horizontal: float
    adaptive_smoothness: float
    crop_percent: float
    min_height: int
    center_analysis_percent: float
    blend_zone_left_start: int
    blend_zone_left_end: int
    blend_zone_right_start: int
    blend_zone_right_end: int
    
    def save(self, filepath: str):
        """Сохранить параметры в файл"""
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
        
        kwargs = {}
        for key, value in data.items():
            if value['type'] == 'numpy':
                kwargs[key] = np.frombuffer(value['data'], dtype=np.dtype(value['dtype'])).reshape(value['shape'])
            elif value['type'] == 'tuple':
                kwargs[key] = tuple(value['data'])
            else:
                kwargs[key] = value['data']
        
        return cls(**kwargs)


class OptimizedCylindricalStitcher3:
    """
    Класс для сшивки трех видео
    """
    
    def __init__(self, video_left_path: str = None, video_center_path: str = None, 
                 video_right_path: str = None, output_path: str = None,
                 num_calibration_frames: int = 10,
                 fov_horizontal: float = 150, adaptive_smoothness: float = 50.0, 
                 crop_percent: float = 0.15):
        
        self.video_left_path = video_left_path
        self.video_center_path = video_center_path
        self.video_right_path = video_right_path
        self.output_path = output_path
        self.num_calibration_frames = min(num_calibration_frames, 5) 
        self.fov_horizontal = fov_horizontal
        self.adaptive_smoothness = adaptive_smoothness
        self.crop_percent = max(0.0, min(0.5, crop_percent))

        # Используем ORB
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Параметры сшивки
        self.homography_left_to_center = None
        self.homography_right_to_center = None
        self.final_transform_left = None
        self.final_transform_center = None
        self.final_transform_right = None
        self.output_size = None
        self.blend_mask_left = None
        self.blend_mask_right = None
        
        # Параметры проекции
        self.projection_map_x = None
        self.projection_map_y = None
        self.adaptive_map_x = None
        self.adaptive_map_y = None
        
        # Параметры для обрезки
        self.top_boundary_smooth = None
        self.bottom_boundary_smooth = None
        self.target_height = None
        self.crop_left = self.crop_top = self.crop_right = self.crop_bottom = 0
        self.final_output_size = (640, 480)  # Размер по умолчанию
        
        # Параметры blending зон
        self.blend_zone_left_start = self.blend_zone_left_end = 0
        self.blend_zone_right_start = self.blend_zone_right_end = 0
        
        self.min_height = 150
        self.center_analysis_percent = 0.8
        
        # Флаг для кэширования
        self._blend_masks_computed = False

    def extract_features_orb(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Извлечение признаков ORB из изображения"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] > 480:
            scale = 480 / gray.shape[0]
            new_w = int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (new_w, 480))
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def find_homography_orb(self, img_src: np.ndarray, img_dst: np.ndarray) -> Optional[np.ndarray]:
        """Поиск матрицы гомографии с использованием ORB"""
        kp_src, desc_src = self.extract_features_orb(img_src)
        kp_dst, desc_dst = self.extract_features_orb(img_dst)

        if desc_src is None or desc_dst is None or len(desc_src) < 8 or len(desc_dst) < 8:
            return None

        # Поиск соответствий
        matches = self.bf.knnMatch(desc_src, desc_dst, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            return None

        src_pts = np.float32([kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def calculate_homographies_from_frames(self) -> None:
        """Вычисление матриц гомографии с ограничением по памяти"""
        logger.info("Вычисление матриц гомографии...")
        
        cap_left = cv2.VideoCapture(self.video_left_path)
        cap_center = cv2.VideoCapture(self.video_center_path)
        cap_right = cv2.VideoCapture(self.video_right_path)

        homographies_left_to_center = []
        homographies_right_to_center = []
        
        for i in range(self.num_calibration_frames):
            ret_left, frame_left = cap_left.read()
            ret_center, frame_center = cap_center.read()
            ret_right, frame_right = cap_right.read()

            if not ret_left or not ret_center or not ret_right:
                break

            # Уменьшаем кадры для поиска гомографии
            h, w = frame_center.shape[:2]
            if h > 360:
                scale = 360 / h
                new_w = int(w * scale)
                frame_left_small = cv2.resize(frame_left, (new_w, 360))
                frame_center_small = cv2.resize(frame_center, (new_w, 360))
                frame_right_small = cv2.resize(frame_right, (new_w, 360))
            else:
                frame_left_small = frame_left
                frame_center_small = frame_center
                frame_right_small = frame_right

            H_left_to_center = self.find_homography_orb(frame_left_small, frame_center_small)
            H_right_to_center = self.find_homography_orb(frame_right_small, frame_center_small)

            if H_left_to_center is not None:
                homographies_left_to_center.append(H_left_to_center)
            if H_right_to_center is not None:
                homographies_right_to_center.append(H_right_to_center)
            
            if (i + 1) % 2 == 0:
                logger.info(f"  Обработано {i + 1}/{self.num_calibration_frames} кадров")

        cap_left.release()
        cap_center.release()
        cap_right.release()

        if homographies_left_to_center:
            self.homography_left_to_center = np.median(homographies_left_to_center, axis=0)
            self.homography_left_to_center = self.homography_left_to_center / self.homography_left_to_center[2, 2]
            logger.info(f"Гомография левого->центрального вычислена")
        else:
            raise Exception("Не удалось вычислить гомографию левого видео")

        if homographies_right_to_center:
            self.homography_right_to_center = np.median(homographies_right_to_center, axis=0)
            self.homography_right_to_center = self.homography_right_to_center / self.homography_right_to_center[2, 2]
            logger.info(f"Гомография правого->центрального вычислена")
        else:
            raise Exception("Не удалось вычислить гомографию правого видео")

    def calculate_final_transforms(self, frame_left: np.ndarray, frame_center: np.ndarray, 
                                   frame_right: np.ndarray) -> None:
        """Вычисление финальных трансформаций"""
        h_c, w_c = frame_center.shape[:2]
        h_l, w_l = frame_left.shape[:2]
        h_r, w_r = frame_right.shape[:2]

        corners_left = np.array([[0, 0], [w_l, 0], [w_l, h_l], [0, h_l]], dtype=np.float32)
        corners_left_transformed = cv2.perspectiveTransform(corners_left.reshape(-1, 1, 2),
                                                            self.homography_left_to_center)

        corners_right = np.array([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]], dtype=np.float32)
        corners_right_transformed = cv2.perspectiveTransform(corners_right.reshape(-1, 1, 2),
                                                             self.homography_right_to_center)

        corners_center = np.array([[0, 0], [w_c, 0], [w_c, h_c], [0, h_c]], dtype=np.float32).reshape(-1, 1, 2)

        all_corners = np.vstack([corners_left_transformed, corners_center, corners_right_transformed])
        min_x, min_y = np.min(all_corners[:, 0, :], axis=0)
        max_x, max_y = np.max(all_corners[:, 0, :], axis=0)

        padding_x = 50
        padding_y = 400 #Костыль
        total_width = int(max_x - min_x) + 2 * padding_x
        total_height = int(max_y - min_y) + 2 * padding_y

        # Ограничиваем размер панорамы
        max_width = 3840
        max_height = 1920
        if total_width > max_width:
            scale = max_width / total_width
            total_width = max_width
            total_height = int(total_height * scale)
        
        if total_height > max_height:
            scale = max_height / total_height
            total_height = max_height
            total_width = int(total_width * scale)

        translation = np.array([[1, 0, -min_x + padding_x],
                                [0, 1, -min_y + padding_y],
                                [0, 0, 1]])

        self.final_transform_left = translation @ self.homography_left_to_center
        self.final_transform_center = translation
        self.final_transform_right = translation @ self.homography_right_to_center
        self.output_size = (total_width, total_height)

        logger.info(f"Размер панорамы: {total_width}x{total_height}")

    def precompute_blend_masks_optimized(self, frame_left: np.ndarray, frame_center: np.ndarray, 
                                         frame_right: np.ndarray) -> None:
        """Вычисление масок blending"""
        if self._blend_masks_computed:
            return
            
        h, w = self.output_size[1], self.output_size[0]
        
        # Используем уменьшенные кадры для вычисления масок
        scale = min(640 / max(frame_center.shape[0], frame_center.shape[1]), 1.0)
        if scale < 1.0:
            small_center = cv2.resize(frame_center, (int(frame_center.shape[1] * scale), 
                                                     int(frame_center.shape[0] * scale)))
            small_left = cv2.resize(frame_left, (int(frame_left.shape[1] * scale), 
                                                 int(frame_left.shape[0] * scale)))
            small_right = cv2.resize(frame_right, (int(frame_right.shape[1] * scale), 
                                                   int(frame_right.shape[0] * scale)))
        else:
            small_center = frame_center
            small_left = frame_left
            small_right = frame_right

        # Вычисляем трансформации в уменьшенном масштабе
        scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        inv_scale_matrix = np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]])
        
        final_transform_left_small = scale_matrix @ self.final_transform_left @ inv_scale_matrix
        final_transform_center_small = scale_matrix @ self.final_transform_center @ inv_scale_matrix
        final_transform_right_small = scale_matrix @ self.final_transform_right @ inv_scale_matrix
        
        small_w = int(self.output_size[0] * scale)
        small_h = int(self.output_size[1] * scale)
        
        warped_center_small = cv2.warpPerspective(small_center, final_transform_center_small, 
                                                   (small_w, small_h),
                                                   flags=cv2.INTER_LINEAR, 
                                                   borderMode=cv2.BORDER_CONSTANT)
        warped_left_small = cv2.warpPerspective(small_left, final_transform_left_small, 
                                                 (small_w, small_h),
                                                 flags=cv2.INTER_LINEAR, 
                                                 borderMode=cv2.BORDER_CONSTANT)
        warped_right_small = cv2.warpPerspective(small_right, final_transform_right_small, 
                                                  (small_w, small_h),
                                                  flags=cv2.INTER_LINEAR, 
                                                  borderMode=cv2.BORDER_CONSTANT)

        mask_center_small = (warped_center_small.sum(axis=2) > 10)
        mask_left_small = (warped_left_small.sum(axis=2) > 10)
        mask_right_small = (warped_right_small.sum(axis=2) > 10)

        overlap_left = mask_left_small & mask_center_small
        overlap_right = mask_right_small & mask_center_small

        self.blend_mask_left = np.zeros((h, w), dtype=np.float32)
        self.blend_mask_right = np.zeros((h, w), dtype=np.float32)

        # Определяем зоны blending
        if np.any(overlap_left):
            overlap_cols = np.where(np.any(overlap_left, axis=0))[0]
            if len(overlap_cols) > 0:
                self.blend_zone_left_start = int(overlap_cols[0] / scale)
                self.blend_zone_left_end = int(overlap_cols[-1] / scale)
                self._create_blend_mask(self.blend_mask_left, self.blend_zone_left_start, 
                                        self.blend_zone_left_end, w, increasing=True)

        if np.any(overlap_right):
            overlap_cols = np.where(np.any(overlap_right, axis=0))[0]
            if len(overlap_cols) > 0:
                self.blend_zone_right_start = int(overlap_cols[0] / scale)
                self.blend_zone_right_end = int(overlap_cols[-1] / scale)
                self._create_blend_mask(self.blend_mask_right, self.blend_zone_right_start, 
                                        self.blend_zone_right_end, w, increasing=False)

        self._blend_masks_computed = True
        logger.info(f"Blending зоны: левая [{self.blend_zone_left_start}-{self.blend_zone_left_end}], "
                   f"правая [{self.blend_zone_right_start}-{self.blend_zone_right_end}]")

    def _create_blend_mask(self, mask: np.ndarray, start: int, end: int, width: int, increasing: bool):
        """Создание маски blending"""
        start = max(0, min(start, width))
        end = max(start + 1, min(end, width))
        
        overlap_width = end - start
        if overlap_width > 0:
            for x in range(start, end):
                t = (x - start) / overlap_width
                alpha = 1 / (1 + np.exp(-12 * (t - 0.5)))
                if not increasing:
                    alpha = 1 - alpha
                mask[:, x] = alpha

    def stitch_frame_optimized(self, frame_left: np.ndarray, frame_center: np.ndarray, 
                               frame_right: np.ndarray) -> np.ndarray:
        """Сшивка кадра"""
        warped_left = cv2.warpPerspective(frame_left, self.final_transform_left, self.output_size,
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped_center = cv2.warpPerspective(frame_center, self.final_transform_center, self.output_size,
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped_right = cv2.warpPerspective(frame_right, self.final_transform_right, self.output_size,
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        result = warped_center.copy()
        
        mask_center = (warped_center.sum(axis=2) > 10)
        mask_left = (warped_left.sum(axis=2) > 10)
        mask_right = (warped_right.sum(axis=2) > 10)

        # Левое видео
        overlap_left = mask_left & mask_center
        if np.any(overlap_left):
            blend_mask_3d = np.stack([self.blend_mask_left] * 3, axis=2)
            result[overlap_left] = (warped_left[overlap_left] * (1 - blend_mask_3d[overlap_left]) + 
                                   warped_center[overlap_left] * blend_mask_3d[overlap_left]).astype(np.uint8)
        
        left_only = mask_left & ~mask_center
        result[left_only] = warped_left[left_only]

        # Правое видео
        overlap_right = mask_right & mask_center
        if np.any(overlap_right):
            blend_mask_3d = np.stack([self.blend_mask_right] * 3, axis=2)
            result[overlap_right] = (warped_right[overlap_right] * (1 - blend_mask_3d[overlap_right]) + 
                                    warped_center[overlap_right] * blend_mask_3d[overlap_right]).astype(np.uint8)
        
        right_only = mask_right & ~mask_center & ~mask_left
        result[right_only] = warped_right[right_only]

        return result

    def create_cylindrical_map(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Создание LUT для цилиндрической проекции"""
        f = width / (2 * math.tan(math.radians(self.fov_horizontal / 2)))
        
        x = np.arange(width, dtype=np.float32)
        y = np.arange(height, dtype=np.float32)
        
        center_x = width / 2
        center_y = height / 2
        
        X, Y = np.meshgrid(x, y)
        
        theta = (X - center_x) / f
        h_cyl = (Y - center_y) / f
        
        map_x = f * np.tan(theta) + center_x
        map_y = f * h_cyl / np.cos(theta) + center_y
        
        map_x = np.clip(np.nan_to_num(map_x, nan=0.0, posinf=width-1, neginf=0.0), 0, width - 1)
        map_y = np.clip(np.nan_to_num(map_y, nan=0.0, posinf=height-1, neginf=0.0), 0, height - 1)
        
        return map_x.astype(np.float32), map_y.astype(np.float32)

    def apply_projection(self, frame: np.ndarray) -> np.ndarray:
        """Применение цилиндрической проекции"""
        if self.projection_map_x is None or self.projection_map_y is None:
            raise ValueError("Карты проекции не инициализированы")
        
        return cv2.remap(frame, self.projection_map_x, self.projection_map_y,
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def remove_black_borders(self, image: np.ndarray, threshold: int = 10) -> Tuple[int, int, int, int]:
        """Удаление черных границ"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mask = gray > threshold
        
        if not np.any(mask):
            return 0, 0, w, h
        
        col_sums = np.sum(mask, axis=0)
        row_sums = np.sum(mask, axis=1)
        
        left = np.argmax(col_sums > 0)
        right = w - 1 - np.argmax(col_sums[::-1] > 0)
        top = np.argmax(row_sums > 0)
        bottom = h - 1 - np.argmax(row_sums[::-1] > 0)
        
        margin = 2
        return (max(0, left - margin), max(0, top - margin), 
                min(w, right + margin), min(h, bottom + margin))

    def apply_side_crop(self, image: np.ndarray) -> np.ndarray:
        """Обрезка боковых артефактов"""
        h, w = image.shape[:2]
        crop_pixels = int(w * self.crop_percent)
        
        if crop_pixels > 0 and w > 2 * crop_pixels:
            return image[:, crop_pixels:w - crop_pixels]
        return image

    def analyze_panorama_boundaries(self, panorama: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Анализ границ панорамы"""
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

    def compute_target_height(self, top_boundary: np.ndarray, bottom_boundary: np.ndarray, w: int) -> int:
        """Вычисление целевой высоты"""
        center_start = int(w * (1 - self.center_analysis_percent) / 2)
        center_end = int(w * (1 + self.center_analysis_percent) / 2)
        
        target_top = np.max(top_boundary[center_start:center_end])
        target_bottom = np.min(bottom_boundary[center_start:center_end])
        target_height = int(target_bottom - target_top)
        
        return max(target_height, self.min_height)

    def smooth_boundaries(self, top_boundary: np.ndarray, bottom_boundary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Сглаживание границ"""
        sigma = self.adaptive_smoothness / 10.0
        top_smooth = gaussian_filter1d(top_boundary, sigma=sigma, mode='reflect')
        bottom_smooth = gaussian_filter1d(bottom_boundary, sigma=sigma, mode='reflect')
        
        top_smooth = np.clip(top_smooth, 0, len(top_boundary)-1)
        bottom_smooth = np.clip(bottom_smooth, 0, len(bottom_boundary)-1)
        
        return top_smooth, bottom_smooth

    def create_adaptive_height_map(self, panorama: np.ndarray, top_smooth: np.ndarray, 
                                   bottom_smooth: np.ndarray, target_height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Создание карты адаптивного масштабирования"""
        h, w = panorama.shape[:2]
        
        map_x = np.zeros((target_height, w), dtype=np.float32)
        map_y = np.zeros((target_height, w), dtype=np.float32)
        
        for col in range(w):
            current_top = top_smooth[col]
            current_bottom = bottom_smooth[col]
            current_height = max(current_bottom - current_top, 1)
            scale = target_height / current_height
            
            for row in range(target_height):
                src_y = current_top + (row / scale)
                src_y = np.clip(src_y, 0, h-1)
                map_x[row, col] = col
                map_y[row, col] = src_y
        
        return map_x, map_y

    def apply_adaptive_scaling(self, panorama: np.ndarray) -> np.ndarray:
        """Применение адаптивного масштабирования"""
        h, w = panorama.shape[:2]
        
        top, bottom = self.analyze_panorama_boundaries(panorama)
        target_height = self.compute_target_height(top, bottom, w)
        
        if target_height % 2 != 0:
            target_height += 1
        
        top_smooth, bottom_smooth = self.smooth_boundaries(top, bottom)
        
        self.top_boundary_smooth = top_smooth
        self.bottom_boundary_smooth = bottom_smooth
        self.target_height = target_height
        
        map_x, map_y = self.create_adaptive_height_map(panorama, top_smooth, bottom_smooth, target_height)
        self.adaptive_map_x = map_x
        self.adaptive_map_y = map_y
        
        return cv2.remap(panorama, map_x.astype(np.float32), map_y.astype(np.float32), 
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def analyze_and_compute_crop_params(self, stitched_frame: np.ndarray) -> None:
        """Анализ и вычисление параметров обрезки"""
        logger.info("Анализ параметров обработки...")
        
        projected = self.apply_projection(stitched_frame)
        side_cropped = self.apply_side_crop(projected)
        left, top, right, bottom = self.remove_black_borders(side_cropped)
        
        self.crop_left, self.crop_top, self.crop_right, self.crop_bottom = left, top, right, bottom
        
        borders_removed = side_cropped[top:bottom, left:right]
        scaled = self.apply_adaptive_scaling(borders_removed)
        
        self.final_output_size = (scaled.shape[1] if scaled.shape[1] % 2 == 0 else scaled.shape[1] - 1,
                                  scaled.shape[0] if scaled.shape[0] % 2 == 0 else scaled.shape[0] - 1)
        
        logger.info(f"Финальный размер: {self.final_output_size[0]}x{self.final_output_size[1]}")

    def process_frame_full_pipeline(self, frame: np.ndarray) -> np.ndarray:
        """Полный пайплайн обработки"""
        projected = self.apply_projection(frame)
        side_cropped = self.apply_side_crop(projected)
        borders_removed = side_cropped[self.crop_top:self.crop_bottom, self.crop_left:self.crop_right]
        
        if self.adaptive_map_x is not None:
            scaled = cv2.remap(borders_removed, self.adaptive_map_x.astype(np.float32),
                              self.adaptive_map_y.astype(np.float32), cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            scaled = self.apply_adaptive_scaling(borders_removed)
        
        if (scaled.shape[1], scaled.shape[0]) != self.final_output_size:
            scaled = cv2.resize(scaled, self.final_output_size)
        
        return scaled

    def initialize_stitching_parameters(self) -> None:
        """Инициализация параметров сшивки"""
        logger.info("Инициализация параметров сшивки...")
        
        self.calculate_homographies_from_frames()

        cap_left = cv2.VideoCapture(self.video_left_path)
        cap_center = cv2.VideoCapture(self.video_center_path)
        cap_right = cv2.VideoCapture(self.video_right_path)

        ret_left, frame_left = cap_left.read()
        ret_center, frame_center = cap_center.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_center or not ret_right:
            raise Exception("Не удалось прочитать кадры для инициализации")

        self.calculate_final_transforms(frame_left, frame_center, frame_right)
        self.precompute_blend_masks_optimized(frame_left, frame_center, frame_right)

        cap_left.release()
        cap_center.release()
        cap_right.release()

        logger.info("Параметры сшивки инициализированы!")

    def get_parameters(self) -> StitchingParameters3:
        """Получить параметры для сохранения"""
        return StitchingParameters3(
            homography_left_to_center=self.homography_left_to_center,
            homography_right_to_center=self.homography_right_to_center,
            final_transform_left=self.final_transform_left,
            final_transform_center=self.final_transform_center,
            final_transform_right=self.final_transform_right,
            output_size=self.output_size,
            blend_mask_left=self.blend_mask_left,
            blend_mask_right=self.blend_mask_right,
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
            center_analysis_percent=self.center_analysis_percent,
            blend_zone_left_start=self.blend_zone_left_start,
            blend_zone_left_end=self.blend_zone_left_end,
            blend_zone_right_start=self.blend_zone_right_start,
            blend_zone_right_end=self.blend_zone_right_end
        )

    def set_parameters(self, params: StitchingParameters3):
        """Установить параметры из загруженного объекта"""
        for key, value in asdict(params).items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._blend_masks_computed = True

    def calibrate(self, calibration_file: str) -> None:
        """Выполнить калибровку"""
        self.initialize_stitching_parameters()
        
        cap_left = cv2.VideoCapture(self.video_left_path)
        cap_center = cv2.VideoCapture(self.video_center_path)
        cap_right = cv2.VideoCapture(self.video_right_path)
        
        ret_left, frame_left = cap_left.read()
        ret_center, frame_center = cap_center.read()
        ret_right, frame_right = cap_right.read()
        
        if not ret_left or not ret_center or not ret_right:
            raise Exception("Не удалось прочитать кадры для калибровки")
        
        first_stitched = self.stitch_frame_optimized(frame_left, frame_center, frame_right)
        
        self.projection_map_x, self.projection_map_y = self.create_cylindrical_map(
            self.output_size[0], self.output_size[1])
        
        self.analyze_and_compute_crop_params(first_stitched)
        
        params = self.get_parameters()
        params.save(calibration_file)
        
        cap_left.release()
        cap_center.release()
        cap_right.release()
        
        logger.info(f"Калибровка завершена. Параметры сохранены в {calibration_file}")

    def process_with_params(self, frame_left: np.ndarray, frame_center: np.ndarray, 
                           frame_right: np.ndarray) -> np.ndarray:
        """Обработка трех кадров с использованием загруженных параметров"""
        stitched = self.stitch_frame_optimized(frame_left, frame_center, frame_right)
        processed = self.process_frame_full_pipeline(stitched)
        return processed
