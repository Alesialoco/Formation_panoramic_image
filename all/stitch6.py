import cv2
import numpy as np
from typing import Tuple, List, Optional
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
class StitchingParameters6:
    """Класс для хранения всех параметров сшивки для 6 камер (360° панорама)"""
    # Гомографии между соседними видео
    homography_1_to_2: np.ndarray
    homography_2_to_3: np.ndarray
    homography_3_to_4: np.ndarray
    homography_4_to_5: np.ndarray
    homography_5_to_6: np.ndarray
    homography_6_to_1: np.ndarray
    
    homography_2_to_1: np.ndarray
    homography_3_to_2: np.ndarray
    homography_4_to_3: np.ndarray
    homography_5_to_4: np.ndarray
    homography_6_to_5: np.ndarray
    homography_1_to_6: np.ndarray
    
    neutral_plane_t: float
    neutral_transform_3_to_neutral: np.ndarray
    neutral_transform_4_to_neutral: np.ndarray
    
    transform_1_to_neutral: np.ndarray
    transform_2_to_neutral: np.ndarray
    transform_3_to_neutral: np.ndarray
    transform_4_to_neutral: np.ndarray
    transform_5_to_neutral: np.ndarray
    transform_6_to_neutral: np.ndarray
    
    final_transform_1: np.ndarray
    final_transform_2: np.ndarray
    final_transform_3: np.ndarray
    final_transform_4: np.ndarray
    final_transform_5: np.ndarray
    final_transform_6: np.ndarray
    output_size: Tuple[int, int]
    
    blend_mask_12: np.ndarray
    blend_mask_23: np.ndarray
    blend_mask_34: np.ndarray
    blend_mask_45: np.ndarray
    blend_mask_56: np.ndarray
    blend_mask_61: np.ndarray
    
    cylinder_radius: float
    full_circle_angle: float
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
    
    def save(self, filepath: str):
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
    def load(cls, filepath: str) -> 'StitchingParameters6':
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


class OptimizedCylindricalStitcher6:
    """
    Класс для сшивки шести видео в панораму 360 градусов
    """
    
    def __init__(self, video1_path: str = None, video2_path: str = None, 
                 video3_path: str = None, video4_path: str = None, 
                 video5_path: str = None, video6_path: str = None,
                 output_path: str = None, num_calibration_frames: int = 10,
                 neutral_plane_t: float = 0.5, fov_horizontal: float = 360,
                 adaptive_smoothness: float = 50.0, crop_percent: float = 0.15):
        
        self.video_paths = [
            video1_path, video2_path, video3_path,
            video4_path, video5_path, video6_path
        ]
        self.output_path = output_path
        self.num_calibration_frames = min(num_calibration_frames, 5)
        self.neutral_plane_t = neutral_plane_t
        self.fov_horizontal = fov_horizontal
        self.adaptive_smoothness = adaptive_smoothness
        self.crop_percent = max(0.0, min(0.5, crop_percent))

        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Инициализация всех гомографий как None
        self.homographies = {}
        for i in range(1, 7):
            next_i = i + 1 if i + 1 <= 6 else 1
            prev_i = i - 1 if i - 1 >= 1 else 6
            self.homographies[f'{i}_to_{next_i}'] = None
            self.homographies[f'{i}_to_{prev_i}'] = None
        
        self.neutral_transform_3_to_neutral = None
        self.neutral_transform_4_to_neutral = None
        
        self.transforms_to_neutral = [None] * 6
        self.final_transforms = [None] * 6
        self.output_size = None
        
        self.blend_masks = [None] * 6  # 6 стыков
        
        self.cylinder_radius = None
        self.full_circle_angle = 360.0
        self.projection_map_x = None
        self.projection_map_y = None
        self.adaptive_map_x = None
        self.adaptive_map_y = None
        
        self.top_boundary_smooth = None
        self.bottom_boundary_smooth = None
        self.target_height = None
        self.crop_left = self.crop_top = self.crop_right = self.crop_bottom = 0
        self.final_output_size = (640, 480)
        
        self.min_height = 150
        self.center_analysis_percent = 0.8
        self._blend_masks_computed = False

    def extract_features_orb(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] > 480:
            scale = 480 / gray.shape[0]
            new_w = int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (new_w, 480))
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def find_homography_orb(self, img_src: np.ndarray, img_dst: np.ndarray) -> Optional[np.ndarray]:
        kp_src, desc_src = self.extract_features_orb(img_src)
        kp_dst, desc_dst = self.extract_features_orb(img_dst)

        if desc_src is None or desc_dst is None or len(desc_src) < 8 or len(desc_dst) < 8:
            return None

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

    def calculate_all_homographies(self) -> bool:
        """Вычисление всех гомографий для замкнутого контура из 6 камер"""
        logger.info("Вычисление матриц гомографии для 360° панорамы...")
        
        # Открываем все видеопотоки
        caps = []
        for path in self.video_paths:
            if path is None:
                logger.error(f"Видео путь не указан")
                return False
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logger.error(f"Не удалось открыть видео: {path}")
                return False
            caps.append(cap)
        
        # Собираем гомографии для каждой пары
        homography_lists = {}
        for i in range(1, 7):
            next_i = i + 1 if i + 1 <= 6 else 1
            prev_i = i - 1 if i - 1 >= 1 else 6
            homography_lists[f'{i}_to_{next_i}'] = []
            homography_lists[f'{i}_to_{prev_i}'] = []
        
        successful_frames = 0
        
        for frame_idx in range(self.num_calibration_frames):
            frames = []
            all_ok = True
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    all_ok = False
                    break
                frames.append(frame)
            
            if not all_ok:
                logger.warning(f"Не удалось прочитать кадр {frame_idx + 1}")
                break
            
            # Уменьшаем кадры для ускорения
            h, w = frames[2].shape[:2]
            if h > 360:
                scale = 360 / h
                new_w = int(w * scale)
                frames_small = [cv2.resize(f, (new_w, 360)) for f in frames]
            else:
                frames_small = frames
            
            # Вычисляем гомографии для всех соседних пар
            for i in range(6):
                next_i = (i + 1) % 6
                H_forward = self.find_homography_orb(frames_small[i], frames_small[next_i])
                H_backward = self.find_homography_orb(frames_small[next_i], frames_small[i])
                
                if H_forward is not None:
                    homography_lists[f'{i+1}_to_{next_i+1}'].append(H_forward)
                if H_backward is not None:
                    homography_lists[f'{next_i+1}_to_{i+1}'].append(H_backward)
            
            successful_frames += 1
            
            if (frame_idx + 1) % 2 == 0:
                logger.info(f"  Обработано {frame_idx + 1}/{self.num_calibration_frames} кадров")
        
        # Закрываем все потоки
        for cap in caps:
            cap.release()
        
        if successful_frames == 0:
            logger.error("Не удалось получить ни одного валидного кадра для калибровки")
            return False
        
        # Сохраняем медианные гомографии
        missing_homographies = []
        for key, lst in homography_lists.items():
            if lst and len(lst) > 0:
                H = np.median(lst, axis=0)
                H = H / H[2, 2]
                self.homographies[key] = H
                logger.info(f"Гомография {key} вычислена (по {len(lst)} кадрам)")
            else:
                missing_homographies.append(key)
                logger.warning(f"Не удалось вычислить гомографию {key}")
        
        if missing_homographies:
            logger.error(f"Отсутствуют гомографии: {missing_homographies}")
            return False
        
        # Устанавливаем атрибуты для доступа
        for key, value in self.homographies.items():
            setattr(self, f'homography_{key}', value)
        
        return True

    def compute_neutral_plane_transforms(self) -> bool:
        """Вычисление преобразований к нейтральной плоскости между видео3 и видео4"""
        logger.info(f"Создание преобразований к нейтральной плоскости (t={self.neutral_plane_t})...")

        # Проверяем наличие необходимых гомографий
        if self.homographies.get('3_to_4') is None:
            logger.error("Отсутствует гомография 3->4")
            return False
        if self.homographies.get('4_to_3') is None:
            logger.error("Отсутствует гомография 4->3")
            return False

        H_3_to_4 = self.homographies['3_to_4']
        H_4_to_3 = self.homographies['4_to_3']

        # Нейтральная плоскость для видео3
        H_3_neutral = (np.eye(3) * (1 - self.neutral_plane_t) + H_3_to_4 * self.neutral_plane_t)
        H_3_neutral = H_3_neutral / H_3_neutral[2, 2]
        
        # Нейтральная плоскость для видео4
        H_4_neutral = (np.eye(3) * self.neutral_plane_t + H_4_to_3 * (1 - self.neutral_plane_t))
        H_4_neutral = H_4_neutral / H_4_neutral[2, 2]

        self.neutral_transform_3_to_neutral = H_3_neutral
        self.neutral_transform_4_to_neutral = H_4_neutral

        logger.info("Преобразования к нейтральной плоскости вычислены")
        return True

    def compute_all_transforms_to_neutral(self) -> bool:
        """Вычисление трансформаций всех видео к нейтральной плоскости"""
        logger.info("Вычисление трансформаций всех видео к нейтральной плоскости...")
        
        # Проверяем наличие всех необходимых гомографий
        required = ['1_to_2', '2_to_3', '4_to_5', '5_to_6']
        for req in required:
            if self.homographies.get(req) is None:
                logger.error(f"Отсутствует гомография {req}")
                return False
        
        # Трансформации к нейтральной плоскости
        # видео1 -> видео2 -> видео3 -> нейтральная
        self.transforms_to_neutral[0] = self.neutral_transform_3_to_neutral @ \
                                         self.homographies['2_to_3'] @ \
                                         self.homographies['1_to_2']
        
        # видео2 -> видео3 -> нейтральная
        self.transforms_to_neutral[1] = self.neutral_transform_3_to_neutral @ \
                                         self.homographies['2_to_3']
        
        # видео3 -> нейтральная
        self.transforms_to_neutral[2] = self.neutral_transform_3_to_neutral
        
        # видео4 -> нейтральная
        self.transforms_to_neutral[3] = self.neutral_transform_4_to_neutral
        
        # видео5 -> видео4 -> нейтральная
        self.transforms_to_neutral[4] = self.neutral_transform_4_to_neutral @ \
                                         self.homographies['4_to_5']
        
        # видео6 -> видео5 -> видео4 -> нейтральная
        self.transforms_to_neutral[5] = self.neutral_transform_4_to_neutral @ \
                                         self.homographies['4_to_5'] @ \
                                         self.homographies['5_to_6']
        
        # Сохраняем как атрибуты
        for i in range(1, 7):
            setattr(self, f'transform_{i}_to_neutral', self.transforms_to_neutral[i-1])
        
        logger.info("Трансформации к нейтральной плоскости вычислены")
        return True

    def calculate_final_transforms(self, frames: list) -> None:
        """Вычисление финальных трансформаций для панорамы 360°"""
        logger.info("Вычисление финальных трансформаций...")
        
        all_corners = []
        
        for frame, transform in zip(frames, self.transforms_to_neutral):
            if transform is None:
                continue
            h, w = frame.shape[:2]
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            corners_transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), transform)
            all_corners.append(corners_transformed)
        
        if len(all_corners) == 0:
            raise ValueError("Нет валидных трансформаций для вычисления размера панорамы")
        
        all_corners = np.vstack(all_corners)
        min_x, min_y = np.min(all_corners[:, 0, :], axis=0)
        max_x, max_y = np.max(all_corners[:, 0, :], axis=0)
        
        padding_x = 150
        padding_y = 150
        total_width = int(max_x - min_x) + 2 * padding_x
        total_height = int(max_y - min_y) + 2 * padding_y
        
        max_width = 10000
        max_height = 4000
        if total_width > max_width:
            scale = max_width / total_width
            total_width = max_width
            total_height = int(total_height * scale)
        
        translation = np.array([[1, 0, -min_x + padding_x],
                                [0, 1, -min_y + padding_y],
                                [0, 0, 1]])
        
        for i in range(6):
            if self.transforms_to_neutral[i] is not None:
                self.final_transforms[i] = translation @ self.transforms_to_neutral[i]
                setattr(self, f'final_transform_{i+1}', self.final_transforms[i])
        
        self.output_size = (total_width, total_height)
        logger.info(f"Размер 360° панорамы: {total_width}x{total_height}")

    def precompute_blend_masks(self, frames: list) -> None:
        """Вычисление масок blending для всех 6 стыков"""
        if self._blend_masks_computed:
            return
            
        h, w = self.output_size[1], self.output_size[0]
        
        scale = min(640 / max(frames[2].shape[0], frames[2].shape[1]), 1.0)
        if scale < 1.0:
            small_frames = [cv2.resize(f, (int(f.shape[1] * scale), int(f.shape[0] * scale))) for f in frames]
        else:
            small_frames = frames
        
        scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        inv_scale_matrix = np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]])
        
        final_transforms_small = []
        for transform in self.final_transforms:
            if transform is not None:
                final_transforms_small.append(scale_matrix @ transform @ inv_scale_matrix)
            else:
                final_transforms_small.append(None)
        
        small_w = int(self.output_size[0] * scale)
        small_h = int(self.output_size[1] * scale)
        
        warped = []
        for frame, transform in zip(small_frames, final_transforms_small):
            if transform is not None:
                warped.append(cv2.warpPerspective(frame, transform, (small_w, small_h),
                                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
            else:
                warped.append(np.zeros((small_h, small_w, 3), dtype=np.uint8))
        
        masks = [(w.sum(axis=2) > 10) for w in warped]
        
        # Стыки: 1-2, 2-3, 3-4, 4-5, 5-6, 6-1
        overlaps = [
            masks[0] & masks[1], masks[1] & masks[2], masks[2] & masks[3],
            masks[3] & masks[4], masks[4] & masks[5], masks[5] & masks[0]
        ]
        
        # Инициализация масок
        for i in range(6):
            self.blend_masks[i] = np.zeros((h, w), dtype=np.float32)
        
        # Направления blending
        directions = [True, True, True, False, False, False]
        
        for idx, (overlap, increasing) in enumerate(zip(overlaps, directions)):
            if np.any(overlap):
                overlap_cols = np.where(np.any(overlap, axis=0))[0]
                if len(overlap_cols) > 0:
                    start = int(overlap_cols[0] / scale)
                    end = int(overlap_cols[-1] / scale)
                    self._create_blend_mask(self.blend_masks[idx], start, end, w, increasing)
                    logger.info(f"Blending зона {idx+1}-{(idx+1)%6+1}: {start}-{end}")
        
        # Сохраняем как атрибуты
        self.blend_mask_12 = self.blend_masks[0]
        self.blend_mask_23 = self.blend_masks[1]
        self.blend_mask_34 = self.blend_masks[2]
        self.blend_mask_45 = self.blend_masks[3]
        self.blend_mask_56 = self.blend_masks[4]
        self.blend_mask_61 = self.blend_masks[5]
        
        self._blend_masks_computed = True

    def _create_blend_mask(self, mask: np.ndarray, start: int, end: int, width: int, increasing: bool):
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

    def create_cylindrical_map_360(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Создание LUT для цилиндрической проекции на 360 градусов"""
        self.cylinder_radius = width / (2 * math.pi)
        
        logger.info(f"Создание цилиндрической проекции для 360° панорамы")
        logger.info(f"  Радиус цилиндра: {self.cylinder_radius:.1f} пикселей")
        
        x = np.arange(width, dtype=np.float32)
        y = np.arange(height, dtype=np.float32)
        
        center_x = width / 2
        center_y = height / 2
        
        X, Y = np.meshgrid(x, y)
        
        theta = 2 * math.pi * (X - center_x) / width
        
        map_x = self.cylinder_radius * np.tan(theta) + center_x
        map_y = self.cylinder_radius * (Y - center_y) / np.cos(theta) + center_y
        
        map_x = np.clip(np.nan_to_num(map_x, nan=0.0, posinf=width-1, neginf=0.0), 0, width - 1)
        map_y = np.clip(np.nan_to_num(map_y, nan=0.0, posinf=height-1, neginf=0.0), 0, height - 1)
        
        return map_x.astype(np.float32), map_y.astype(np.float32)

    def apply_projection(self, frame: np.ndarray) -> np.ndarray:
        if self.projection_map_x is None or self.projection_map_y is None:
            raise ValueError("Карты проекции не инициализированы")
        return cv2.remap(frame, self.projection_map_x, self.projection_map_y,
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def remove_black_borders(self, image: np.ndarray, threshold: int = 10) -> Tuple[int, int, int, int]:
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
        h, w = image.shape[:2]
        crop_pixels = int(w * self.crop_percent)
        if crop_pixels > 0 and w > 2 * crop_pixels:
            return image[:, crop_pixels:w - crop_pixels]
        return image

    def analyze_panorama_boundaries(self, panorama: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        center_start = int(w * (1 - self.center_analysis_percent) / 2)
        center_end = int(w * (1 + self.center_analysis_percent) / 2)
        
        target_top = np.max(top_boundary[center_start:center_end])
        target_bottom = np.min(bottom_boundary[center_start:center_end])
        target_height = int(target_bottom - target_top)
        
        return max(target_height, self.min_height)

    def smooth_boundaries(self, top_boundary: np.ndarray, bottom_boundary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sigma = self.adaptive_smoothness / 10.0
        top_smooth = gaussian_filter1d(top_boundary, sigma=sigma, mode='wrap')
        bottom_smooth = gaussian_filter1d(bottom_boundary, sigma=sigma, mode='wrap')
        
        top_smooth = np.clip(top_smooth, 0, len(top_boundary)-1)
        bottom_smooth = np.clip(bottom_smooth, 0, len(bottom_boundary)-1)
        
        return top_smooth, bottom_smooth

    def create_adaptive_height_map(self, panorama: np.ndarray, top_smooth: np.ndarray, 
                                    bottom_smooth: np.ndarray, target_height: int) -> Tuple[np.ndarray, np.ndarray]:
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
        logger.info("Анализ параметров обработки для 360° панорамы...")
        
        projected = self.apply_projection(stitched_frame)
        side_cropped = self.apply_side_crop(projected)
        left, top, right, bottom = self.remove_black_borders(side_cropped)
        
        self.crop_left, self.crop_top, self.crop_right, self.crop_bottom = left, top, right, bottom
        
        borders_removed = side_cropped[top:bottom, left:right]
        scaled = self.apply_adaptive_scaling(borders_removed)
        
        self.final_output_size = (scaled.shape[1] if scaled.shape[1] % 2 == 0 else scaled.shape[1] - 1,
                                   scaled.shape[0] if scaled.shape[0] % 2 == 0 else scaled.shape[0] - 1)
        
        logger.info(f"Финальный размер 360° панорамы: {self.final_output_size[0]}x{self.final_output_size[1]}")

    def stitch_frame_optimized(self, frames: list) -> np.ndarray:
        """Сшивка шести кадров в панораму 360°"""
        warped = []
        for i, frame in enumerate(frames):
            if self.final_transforms[i] is not None:
                warped.append(cv2.warpPerspective(frame, self.final_transforms[i], 
                                                  self.output_size, flags=cv2.INTER_LINEAR, 
                                                  borderMode=cv2.BORDER_CONSTANT))
            else:
                warped.append(np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8))
        
        result = warped[2].copy()
        
        masks = [(w.sum(axis=2) > 10) for w in warped]
        
        # Стыки: 1-2, 2-3, 3-4, 4-5, 5-6, 6-1
        pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        
        for idx, (i, j) in enumerate(pairs):
            if self.blend_masks[idx] is None:
                continue
            overlap = masks[i] & masks[j]
            if np.any(overlap):
                blend_3d = np.stack([self.blend_masks[idx]] * 3, axis=2)
                if idx < 3:
                    result[overlap] = (warped[i][overlap] * (1 - blend_3d[overlap]) + 
                                       warped[j][overlap] * blend_3d[overlap]).astype(np.uint8)
                else:
                    result[overlap] = (warped[i][overlap] * blend_3d[overlap] + 
                                       warped[j][overlap] * (1 - blend_3d[overlap])).astype(np.uint8)
        
        # Добавляем уникальные области
        for i in [0, 3, 4, 5]:
            if self.final_transforms[i] is None:
                continue
            only = masks[i] & ~masks[2]
            if np.any(only):
                result[only] = warped[i][only]
        
        return result

    def process_frame_full_pipeline(self, frame: np.ndarray) -> np.ndarray:
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

    def initialize_stitching_parameters(self) -> bool:
        """Инициализация параметров сшивки с проверкой всех шагов"""
        logger.info("Инициализация параметров сшивки для 360° панорамы (6 камер)...")
        
        if not self.calculate_all_homographies():
            logger.error("Ошибка при вычислении гомографий")
            return False
        
        if not self.compute_neutral_plane_transforms():
            logger.error("Ошибка при вычислении нейтральной плоскости")
            return False
        
        if not self.compute_all_transforms_to_neutral():
            logger.error("Ошибка при вычислении трансформаций к нейтральной плоскости")
            return False
        
        # Загружаем первые кадры
        caps = []
        frames = []
        for path in self.video_paths:
            if path is None:
                logger.error("Видео путь не указан")
                return False
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logger.error(f"Не удалось открыть видео: {path}")
                return False
            caps.append(cap)
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Не удалось прочитать первый кадр из: {path}")
                return False
            frames.append(frame)
        
        self.calculate_final_transforms(frames)
        self.precompute_blend_masks(frames)
        
        for cap in caps:
            cap.release()
        
        logger.info("Параметры сшивки для 360° панорамы инициализированы!")
        return True

    def get_parameters(self) -> StitchingParameters6:
        return StitchingParameters6(
            homography_1_to_2=self.homographies.get('1_to_2'),
            homography_2_to_3=self.homographies.get('2_to_3'),
            homography_3_to_4=self.homographies.get('3_to_4'),
            homography_4_to_5=self.homographies.get('4_to_5'),
            homography_5_to_6=self.homographies.get('5_to_6'),
            homography_6_to_1=self.homographies.get('6_to_1'),
            homography_2_to_1=self.homographies.get('2_to_1'),
            homography_3_to_2=self.homographies.get('3_to_2'),
            homography_4_to_3=self.homographies.get('4_to_3'),
            homography_5_to_4=self.homographies.get('5_to_4'),
            homography_6_to_5=self.homographies.get('6_to_5'),
            homography_1_to_6=self.homographies.get('1_to_6'),
            neutral_plane_t=self.neutral_plane_t,
            neutral_transform_3_to_neutral=self.neutral_transform_3_to_neutral,
            neutral_transform_4_to_neutral=self.neutral_transform_4_to_neutral,
            transform_1_to_neutral=self.transforms_to_neutral[0],
            transform_2_to_neutral=self.transforms_to_neutral[1],
            transform_3_to_neutral=self.transforms_to_neutral[2],
            transform_4_to_neutral=self.transforms_to_neutral[3],
            transform_5_to_neutral=self.transforms_to_neutral[4],
            transform_6_to_neutral=self.transforms_to_neutral[5],
            final_transform_1=self.final_transforms[0],
            final_transform_2=self.final_transforms[1],
            final_transform_3=self.final_transforms[2],
            final_transform_4=self.final_transforms[3],
            final_transform_5=self.final_transforms[4],
            final_transform_6=self.final_transforms[5],
            output_size=self.output_size,
            blend_mask_12=self.blend_masks[0],
            blend_mask_23=self.blend_masks[1],
            blend_mask_34=self.blend_masks[2],
            blend_mask_45=self.blend_masks[3],
            blend_mask_56=self.blend_masks[4],
            blend_mask_61=self.blend_masks[5],
            cylinder_radius=self.cylinder_radius,
            full_circle_angle=self.full_circle_angle,
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

    def set_parameters(self, params: StitchingParameters6):
        for key, value in asdict(params).items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Восстанавливаем списки из атрибутов
        self.homographies = {}
        for i in range(1, 7):
            next_i = i + 1 if i + 1 <= 6 else 1
            prev_i = i - 1 if i - 1 >= 1 else 6
            self.homographies[f'{i}_to_{next_i}'] = getattr(self, f'homography_{i}_to_{next_i}', None)
            self.homographies[f'{i}_to_{prev_i}'] = getattr(self, f'homography_{i}_to_{prev_i}', None)
        
        self.transforms_to_neutral = [
            self.transform_1_to_neutral, self.transform_2_to_neutral,
            self.transform_3_to_neutral, self.transform_4_to_neutral,
            self.transform_5_to_neutral, self.transform_6_to_neutral
        ]
        
        self.final_transforms = [
            self.final_transform_1, self.final_transform_2, self.final_transform_3,
            self.final_transform_4, self.final_transform_5, self.final_transform_6
        ]
        
        self.blend_masks = [
            self.blend_mask_12, self.blend_mask_23, self.blend_mask_34,
            self.blend_mask_45, self.blend_mask_56, self.blend_mask_61
        ]
        
        self._blend_masks_computed = True

    def calibrate(self, calibration_file: str) -> None:
        """Выполнить калибровку с проверкой успешности"""
        if not self.initialize_stitching_parameters():
            raise Exception("Не удалось инициализировать параметры сшивки. "
                          "Проверьте, что все 6 видеопотоков доступны и имеют достаточное количество общих特征.")
        
        # Загружаем первые кадры для калибровки проекции
        caps = []
        frames = []
        for path in self.video_paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise Exception(f"Не удалось открыть видео: {path}")
            caps.append(cap)
            ret, frame = cap.read()
            if not ret:
                raise Exception(f"Не удалось прочитать первый кадр из: {path}")
            frames.append(frame)
        
        first_stitched = self.stitch_frame_optimized(frames)
        
        self.projection_map_x, self.projection_map_y = self.create_cylindrical_map_360(
            self.output_size[0], self.output_size[1])
        
        self.analyze_and_compute_crop_params(first_stitched)
        
        params = self.get_parameters()
        params.save(calibration_file)
        
        for cap in caps:
            cap.release()
        
        logger.info(f"Калибровка 360° панорамы завершена. Параметры сохранены в {calibration_file}")

    def process_with_params(self, frame1: np.ndarray, frame2: np.ndarray, frame3: np.ndarray,
                            frame4: np.ndarray, frame5: np.ndarray, frame6: np.ndarray) -> np.ndarray:
        stitched = self.stitch_frame_optimized([frame1, frame2, frame3, frame4, frame5, frame6])
        processed = self.process_frame_full_pipeline(stitched)
        return processed
