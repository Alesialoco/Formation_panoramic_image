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
class StitchingParameters4:
    """Класс для хранения всех параметров сшивки для 4 камер"""
    # Гомографии для приведения к нейтральной плоскости
    homography_1_to_2: np.ndarray  # видео1 -> видео2
    homography_2_to_1: np.ndarray  # видео2 -> видео1
    homography_3_to_2: np.ndarray  # видео3 -> видео2
    homography_2_to_3: np.ndarray  # видео2 -> видео3
    homography_4_to_3: np.ndarray  # видео4 -> видео3
    homography_3_to_4: np.ndarray  # видео3 -> видео4
    
    # Трансформации к нейтральной плоскости (между видео2 и видео3)
    neutral_plane_t: float  # параметр нейтральной плоскости
    neutral_transform_2_to_neutral: np.ndarray  # видео2 -> нейтральная
    neutral_transform_3_to_neutral: np.ndarray  # видео3 -> нейтральная
    
    # Трансформации всех видео к нейтральной плоскости
    transform_1_to_neutral: np.ndarray
    transform_2_to_neutral: np.ndarray
    transform_3_to_neutral: np.ndarray
    transform_4_to_neutral: np.ndarray
    
    # Финальные трансформации с учетом сдвига и размера панорамы
    final_transform_1: np.ndarray
    final_transform_2: np.ndarray
    final_transform_3: np.ndarray
    final_transform_4: np.ndarray
    output_size: Tuple[int, int]
    
    # Маски для blending (для всех стыков: 1-2, 2-3, 3-4)
    blend_mask_12: np.ndarray
    blend_mask_23: np.ndarray
    blend_mask_34: np.ndarray
    
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
    
    # Зоны blending
    blend_zone_12_start: int
    blend_zone_12_end: int
    blend_zone_23_start: int
    blend_zone_23_end: int
    blend_zone_34_start: int
    blend_zone_34_end: int
    
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
    def load(cls, filepath: str) -> 'StitchingParameters4':
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


class OptimizedCylindricalStitcher4:
    """
    Класс для сшивки четырех видео в панораму
    
    Схема сшивки:
    - Видео1 преобразуется к плоскости видео2
    - Видео4 преобразуется к плоскости видео3
    - Видео2 и видео3 преобразуются к нейтральной плоскости между ними
    - Все 4 видео сшиваются в единую панораму
    """
    
    def __init__(self, video1_path: str = None, video2_path: str = None, 
                 video3_path: str = None, video4_path: str = None, output_path: str = None,
                 num_calibration_frames: int = 10, neutral_plane_t: float = 0.5,
                 fov_horizontal: float = 150, adaptive_smoothness: float = 50.0, 
                 crop_percent: float = 0.15):
        
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.video3_path = video3_path
        self.video4_path = video4_path
        self.output_path = output_path
        self.num_calibration_frames = min(num_calibration_frames, 5)
        self.neutral_plane_t = neutral_plane_t
        self.fov_horizontal = fov_horizontal
        self.adaptive_smoothness = adaptive_smoothness
        self.crop_percent = max(0.0, min(0.5, crop_percent))

        if not 0 <= neutral_plane_t <= 1:
            raise ValueError("neutral_plane_t должен быть в диапазоне от 0 до 1")

        # Используем ORB для поиска гомографий
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Параметры сшивки
        self.homography_1_to_2 = None
        self.homography_2_to_1 = None
        self.homography_3_to_2 = None
        self.homography_2_to_3 = None
        self.homography_4_to_3 = None
        self.homography_3_to_4 = None
        
        self.neutral_transform_2_to_neutral = None
        self.neutral_transform_3_to_neutral = None
        
        self.transform_1_to_neutral = None
        self.transform_2_to_neutral = None
        self.transform_3_to_neutral = None
        self.transform_4_to_neutral = None
        
        self.final_transform_1 = None
        self.final_transform_2 = None
        self.final_transform_3 = None
        self.final_transform_4 = None
        self.output_size = None
        
        self.blend_mask_12 = None
        self.blend_mask_23 = None
        self.blend_mask_34 = None
        
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
        self.final_output_size = (640, 480)
        
        # Параметры blending зон
        self.blend_zone_12_start = self.blend_zone_12_end = 0
        self.blend_zone_23_start = self.blend_zone_23_end = 0
        self.blend_zone_34_start = self.blend_zone_34_end = 0
        
        self.min_height = 150
        self.center_analysis_percent = 0.8
        
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
        """
        Вычисление всех матриц гомографии:
        - между видео1 и видео2
        - между видео2 и видео3
        - между видео3 и видео4
        """
        logger.info("Вычисление матриц гомографии...")
        
        cap1 = cv2.VideoCapture(self.video1_path)
        cap2 = cv2.VideoCapture(self.video2_path)
        cap3 = cv2.VideoCapture(self.video3_path)
        cap4 = cv2.VideoCapture(self.video4_path)

        homographies_1_to_2 = []
        homographies_2_to_1 = []
        homographies_2_to_3 = []
        homographies_3_to_2 = []
        homographies_3_to_4 = []
        homographies_4_to_3 = []
        
        for i in range(self.num_calibration_frames):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            ret3, frame3 = cap3.read()
            ret4, frame4 = cap4.read()

            if not ret1 or not ret2 or not ret3 or not ret4:
                break

            # Уменьшаем кадры для поиска гомографии
            h, w = frame2.shape[:2]
            if h > 360:
                scale = 360 / h
                new_w = int(w * scale)
                frame1_small = cv2.resize(frame1, (new_w, 360))
                frame2_small = cv2.resize(frame2, (new_w, 360))
                frame3_small = cv2.resize(frame3, (new_w, 360))
                frame4_small = cv2.resize(frame4, (new_w, 360))
            else:
                frame1_small = frame1
                frame2_small = frame2
                frame3_small = frame3
                frame4_small = frame4

            # Гомографии между парой 1-2
            H_1_to_2 = self.find_homography_orb(frame1_small, frame2_small)
            H_2_to_1 = self.find_homography_orb(frame2_small, frame1_small)
            
            # Гомографии между парой 2-3
            H_2_to_3 = self.find_homography_orb(frame2_small, frame3_small)
            H_3_to_2 = self.find_homography_orb(frame3_small, frame2_small)
            
            # Гомографии между парой 3-4
            H_3_to_4 = self.find_homography_orb(frame3_small, frame4_small)
            H_4_to_3 = self.find_homography_orb(frame4_small, frame3_small)

            if H_1_to_2 is not None:
                homographies_1_to_2.append(H_1_to_2)
            if H_2_to_1 is not None:
                homographies_2_to_1.append(H_2_to_1)
            if H_2_to_3 is not None:
                homographies_2_to_3.append(H_2_to_3)
            if H_3_to_2 is not None:
                homographies_3_to_2.append(H_3_to_2)
            if H_3_to_4 is not None:
                homographies_3_to_4.append(H_3_to_4)
            if H_4_to_3 is not None:
                homographies_4_to_3.append(H_4_to_3)
            
            if (i + 1) % 2 == 0:
                logger.info(f"  Обработано {i + 1}/{self.num_calibration_frames} кадров")

        cap1.release()
        cap2.release()
        cap3.release()
        cap4.release()

        # Сохраняем вычисленные гомографии
        if homographies_1_to_2:
            self.homography_1_to_2 = np.median(homographies_1_to_2, axis=0)
            self.homography_1_to_2 = self.homography_1_to_2 / self.homography_1_to_2[2, 2]
            logger.info("Гомография видео1->видео2 вычислена")
        else:
            raise Exception("Не удалось вычислить гомографию видео1->видео2")

        if homographies_2_to_1:
            self.homography_2_to_1 = np.median(homographies_2_to_1, axis=0)
            self.homography_2_to_1 = self.homography_2_to_1 / self.homography_2_to_1[2, 2]
            logger.info("Гомография видео2->видео1 вычислена")

        if homographies_2_to_3:
            self.homography_2_to_3 = np.median(homographies_2_to_3, axis=0)
            self.homography_2_to_3 = self.homography_2_to_3 / self.homography_2_to_3[2, 2]
            logger.info("Гомография видео2->видео3 вычислена")
        else:
            raise Exception("Не удалось вычислить гомографию видео2->видео3")

        if homographies_3_to_2:
            self.homography_3_to_2 = np.median(homographies_3_to_2, axis=0)
            self.homography_3_to_2 = self.homography_3_to_2 / self.homography_3_to_2[2, 2]
            logger.info("Гомография видео3->видео2 вычислена")

        if homographies_3_to_4:
            self.homography_3_to_4 = np.median(homographies_3_to_4, axis=0)
            self.homography_3_to_4 = self.homography_3_to_4 / self.homography_3_to_4[2, 2]
            logger.info("Гомография видео3->видео4 вычислена")
        else:
            raise Exception("Не удалось вычислить гомографию видео3->видео4")

        if homographies_4_to_3:
            self.homography_4_to_3 = np.median(homographies_4_to_3, axis=0)
            self.homography_4_to_3 = self.homography_4_to_3 / self.homography_4_to_3[2, 2]
            logger.info("Гомография видео4->видео3 вычислена")

    def compute_neutral_plane_transforms(self) -> None:
        """
        Вычисление преобразований к нейтральной плоскости между видео2 и видео3
        """
        logger.info(f"Создание преобразований к нейтральной плоскости (t={self.neutral_plane_t})...")

        # Используем гомографию между видео2 и видео3
        H_2_to_3 = self.homography_2_to_3 / self.homography_2_to_3[2, 2]
        H_3_to_2 = self.homography_3_to_2 / self.homography_3_to_2[2, 2]

        # Нейтральная плоскость для видео2 (приведение к нейтральной)
        H_2_neutral = (np.eye(3) * (1 - self.neutral_plane_t) + H_2_to_3 * self.neutral_plane_t)
        H_2_neutral = H_2_neutral / H_2_neutral[2, 2]
        
        # Нейтральная плоскость для видео3 (приведение к нейтральной)
        H_3_neutral = (np.eye(3) * self.neutral_plane_t + H_3_to_2 * (1 - self.neutral_plane_t))
        H_3_neutral = H_3_neutral / H_3_neutral[2, 2]

        self.neutral_transform_2_to_neutral = H_2_neutral
        self.neutral_transform_3_to_neutral = H_3_neutral

        logger.info("Преобразования к нейтральной плоскости вычислены")

    def compute_all_transforms_to_neutral(self, frame1: np.ndarray, frame2: np.ndarray,
                                           frame3: np.ndarray, frame4: np.ndarray) -> None:
        """
        Вычисление преобразований всех видео к нейтральной плоскости
        
        Видео1 -> преобразуется к видео2 -> затем к нейтральной
        Видео2 -> преобразуется к нейтральной
        Видео3 -> преобразуется к нейтральной
        Видео4 -> преобразуется к видео3 -> затем к нейтральной
        """
        logger.info("Вычисление преобразований всех видео к нейтральной плоскости...")
        
        # Трансформация видео1: сначала к видео2, затем к нейтральной
        self.transform_1_to_neutral = self.neutral_transform_2_to_neutral @ self.homography_1_to_2
        
        # Трансформация видео2: прямо к нейтральной
        self.transform_2_to_neutral = self.neutral_transform_2_to_neutral
        
        # Трансформация видео3: прямо к нейтральной
        self.transform_3_to_neutral = self.neutral_transform_3_to_neutral
        
        # Трансформация видео4: сначала к видео3, затем к нейтральной
        self.transform_4_to_neutral = self.neutral_transform_3_to_neutral @ self.homography_4_to_3
        
        logger.info("Преобразования всех видео к нейтральной плоскости вычислены")

    def calculate_final_transforms(self, frame1: np.ndarray, frame2: np.ndarray,
                                    frame3: np.ndarray, frame4: np.ndarray) -> None:
        """
        Вычисление финальных трансформаций с учетом размера панорамы и сдвига
        """
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        h3, w3 = frame3.shape[:2]
        h4, w4 = frame4.shape[:2]

        # Трансформируем углы всех видео в нейтральную плоскость
        corners1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
        corners1_transformed = cv2.perspectiveTransform(corners1.reshape(-1, 1, 2),
                                                        self.transform_1_to_neutral)

        corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
        corners2_transformed = cv2.perspectiveTransform(corners2.reshape(-1, 1, 2),
                                                        self.transform_2_to_neutral)

        corners3 = np.array([[0, 0], [w3, 0], [w3, h3], [0, h3]], dtype=np.float32)
        corners3_transformed = cv2.perspectiveTransform(corners3.reshape(-1, 1, 2),
                                                        self.transform_3_to_neutral)

        corners4 = np.array([[0, 0], [w4, 0], [w4, h4], [0, h4]], dtype=np.float32)
        corners4_transformed = cv2.perspectiveTransform(corners4.reshape(-1, 1, 2),
                                                        self.transform_4_to_neutral)

        all_corners = np.vstack([corners1_transformed, corners2_transformed,
                                  corners3_transformed, corners4_transformed])
        min_x, min_y = np.min(all_corners[:, 0, :], axis=0)
        max_x, max_y = np.max(all_corners[:, 0, :], axis=0)

        # Добавляем отступы
        padding_x = 100
        padding_y = 100
        total_width = int(max_x - min_x) + 2 * padding_x
        total_height = int(max_y - min_y) + 2 * padding_y

        # Ограничиваем размер панорамы
        max_width = 7680
        max_height = 3840
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

        self.final_transform_1 = translation @ self.transform_1_to_neutral
        self.final_transform_2 = translation @ self.transform_2_to_neutral
        self.final_transform_3 = translation @ self.transform_3_to_neutral
        self.final_transform_4 = translation @ self.transform_4_to_neutral
        self.output_size = (total_width, total_height)

        logger.info(f"Размер панорамы: {total_width}x{total_height}")

    def precompute_blend_masks(self, frame1: np.ndarray, frame2: np.ndarray,
                                frame3: np.ndarray, frame4: np.ndarray) -> None:
        """
        Вычисление масок blending для всех стыков:
        - между видео1 и видео2
        - между видео2 и видео3
        - между видео3 и видео4
        """
        if self._blend_masks_computed:
            return
            
        h, w = self.output_size[1], self.output_size[0]
        
        # Используем уменьшенные кадры для вычисления масок
        scale = min(640 / max(frame2.shape[0], frame2.shape[1]), 1.0)
        if scale < 1.0:
            small1 = cv2.resize(frame1, (int(frame1.shape[1] * scale), int(frame1.shape[0] * scale)))
            small2 = cv2.resize(frame2, (int(frame2.shape[1] * scale), int(frame2.shape[0] * scale)))
            small3 = cv2.resize(frame3, (int(frame3.shape[1] * scale), int(frame3.shape[0] * scale)))
            small4 = cv2.resize(frame4, (int(frame4.shape[1] * scale), int(frame4.shape[0] * scale)))
        else:
            small1, small2, small3, small4 = frame1, frame2, frame3, frame4

        # Масштабируем трансформации
        scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        inv_scale_matrix = np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]])
        
        final_transform_1_small = scale_matrix @ self.final_transform_1 @ inv_scale_matrix
        final_transform_2_small = scale_matrix @ self.final_transform_2 @ inv_scale_matrix
        final_transform_3_small = scale_matrix @ self.final_transform_3 @ inv_scale_matrix
        final_transform_4_small = scale_matrix @ self.final_transform_4 @ inv_scale_matrix
        
        small_w = int(self.output_size[0] * scale)
        small_h = int(self.output_size[1] * scale)
        
        # Варпируем все видео
        warped1 = cv2.warpPerspective(small1, final_transform_1_small, (small_w, small_h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped2 = cv2.warpPerspective(small2, final_transform_2_small, (small_w, small_h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped3 = cv2.warpPerspective(small3, final_transform_3_small, (small_w, small_h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped4 = cv2.warpPerspective(small4, final_transform_4_small, (small_w, small_h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Маски ненулевых пикселей
        mask1 = (warped1.sum(axis=2) > 10)
        mask2 = (warped2.sum(axis=2) > 10)
        mask3 = (warped3.sum(axis=2) > 10)
        mask4 = (warped4.sum(axis=2) > 10)

        # Зоны перекрытия
        overlap_12 = mask1 & mask2
        overlap_23 = mask2 & mask3
        overlap_34 = mask3 & mask4

        self.blend_mask_12 = np.zeros((h, w), dtype=np.float32)
        self.blend_mask_23 = np.zeros((h, w), dtype=np.float32)
        self.blend_mask_34 = np.zeros((h, w), dtype=np.float32)

        # Определяем зоны blending для стыка 1-2
        if np.any(overlap_12):
            overlap_cols = np.where(np.any(overlap_12, axis=0))[0]
            if len(overlap_cols) > 0:
                start = int(overlap_cols[0] / scale)
                end = int(overlap_cols[-1] / scale)
                self.blend_zone_12_start = start
                self.blend_zone_12_end = end
                self._create_blend_mask(self.blend_mask_12, start, end, w, increasing=True)
                logger.info(f"Blending зона 1-2: {start}-{end}")

        # Определяем зоны blending для стыка 2-3
        if np.any(overlap_23):
            overlap_cols = np.where(np.any(overlap_23, axis=0))[0]
            if len(overlap_cols) > 0:
                start = int(overlap_cols[0] / scale)
                end = int(overlap_cols[-1] / scale)
                self.blend_zone_23_start = start
                self.blend_zone_23_end = end
                # Для центрального стыка blending симметричный от центра
                self._create_blend_mask(self.blend_mask_23, start, end, w, increasing=True)
                logger.info(f"Blending зона 2-3: {start}-{end}")

        # Определяем зоны blending для стыка 3-4
        if np.any(overlap_34):
            overlap_cols = np.where(np.any(overlap_34, axis=0))[0]
            if len(overlap_cols) > 0:
                start = int(overlap_cols[0] / scale)
                end = int(overlap_cols[-1] / scale)
                self.blend_zone_34_start = start
                self.blend_zone_34_end = end
                self._create_blend_mask(self.blend_mask_34, start, end, w, increasing=False)
                logger.info(f"Blending зона 3-4: {start}-{end}")

        self._blend_masks_computed = True

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

    def stitch_frame_optimized(self, frame1: np.ndarray, frame2: np.ndarray,
                                frame3: np.ndarray, frame4: np.ndarray) -> np.ndarray:
        """
        Сшивка четырех кадров в панораму
        """
        # Варпируем все видео
        warped1 = cv2.warpPerspective(frame1, self.final_transform_1, self.output_size,
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped2 = cv2.warpPerspective(frame2, self.final_transform_2, self.output_size,
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped3 = cv2.warpPerspective(frame3, self.final_transform_3, self.output_size,
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped4 = cv2.warpPerspective(frame4, self.final_transform_4, self.output_size,
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Результат начинаем с видео2
        result = warped2.copy()
        
        # Маски ненулевых пикселей
        mask1 = (warped1.sum(axis=2) > 10)
        mask2 = (warped2.sum(axis=2) > 10)
        mask3 = (warped3.sum(axis=2) > 10)
        mask4 = (warped4.sum(axis=2) > 10)

        # === Обработка стыка 1-2 ===
        overlap_12 = mask1 & mask2
        if np.any(overlap_12):
            blend_mask_3d = np.stack([self.blend_mask_12] * 3, axis=2)
            result[overlap_12] = (warped1[overlap_12] * (1 - blend_mask_3d[overlap_12]) + 
                                   warped2[overlap_12] * blend_mask_3d[overlap_12]).astype(np.uint8)
        
        # Добавляем области только из видео1
        only1 = mask1 & ~mask2
        result[only1] = warped1[only1]

        # === Обработка стыка 2-3 ===
        overlap_23 = mask2 & mask3
        if np.any(overlap_23):
            blend_mask_3d = np.stack([self.blend_mask_23] * 3, axis=2)
            result[overlap_23] = (warped2[overlap_23] * (1 - blend_mask_3d[overlap_23]) + 
                                   warped3[overlap_23] * blend_mask_3d[overlap_23]).astype(np.uint8)

        # Добавляем области только из видео3 (которые не перекрываются с видео2)
        only3 = mask3 & ~mask2
        result[only3] = warped3[only3]

        # === Обработка стыка 3-4 ===
        overlap_34 = mask3 & mask4
        if np.any(overlap_34):
            blend_mask_3d = np.stack([self.blend_mask_34] * 3, axis=2)
            result[overlap_34] = (warped3[overlap_34] * (1 - blend_mask_3d[overlap_34]) + 
                                   warped4[overlap_34] * blend_mask_3d[overlap_34]).astype(np.uint8)
        
        # Добавляем области только из видео4
        only4 = mask4 & ~mask3
        result[only4] = warped4[only4]

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
        """Инициализация всех параметров сшивки"""
        logger.info("Инициализация параметров сшивки для 4 видео...")
        
        self.calculate_homographies_from_frames()
        self.compute_neutral_plane_transforms()

        # Захватываем первые кадры для инициализации
        cap1 = cv2.VideoCapture(self.video1_path)
        cap2 = cv2.VideoCapture(self.video2_path)
        cap3 = cv2.VideoCapture(self.video3_path)
        cap4 = cv2.VideoCapture(self.video4_path)

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()

        if not ret1 or not ret2 or not ret3 or not ret4:
            raise Exception("Не удалось прочитать кадры для инициализации")

        self.compute_all_transforms_to_neutral(frame1, frame2, frame3, frame4)
        self.calculate_final_transforms(frame1, frame2, frame3, frame4)
        self.precompute_blend_masks(frame1, frame2, frame3, frame4)

        cap1.release()
        cap2.release()
        cap3.release()
        cap4.release()

        logger.info("Параметры сшивки инициализированы!")

    def get_parameters(self) -> StitchingParameters4:
        """Получить параметры для сохранения"""
        return StitchingParameters4(
            homography_1_to_2=self.homography_1_to_2,
            homography_2_to_1=self.homography_2_to_1,
            homography_3_to_2=self.homography_3_to_2,
            homography_2_to_3=self.homography_2_to_3,
            homography_4_to_3=self.homography_4_to_3,
            homography_3_to_4=self.homography_3_to_4,
            neutral_plane_t=self.neutral_plane_t,
            neutral_transform_2_to_neutral=self.neutral_transform_2_to_neutral,
            neutral_transform_3_to_neutral=self.neutral_transform_3_to_neutral,
            transform_1_to_neutral=self.transform_1_to_neutral,
            transform_2_to_neutral=self.transform_2_to_neutral,
            transform_3_to_neutral=self.transform_3_to_neutral,
            transform_4_to_neutral=self.transform_4_to_neutral,
            final_transform_1=self.final_transform_1,
            final_transform_2=self.final_transform_2,
            final_transform_3=self.final_transform_3,
            final_transform_4=self.final_transform_4,
            output_size=self.output_size,
            blend_mask_12=self.blend_mask_12,
            blend_mask_23=self.blend_mask_23,
            blend_mask_34=self.blend_mask_34,
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
            blend_zone_12_start=self.blend_zone_12_start,
            blend_zone_12_end=self.blend_zone_12_end,
            blend_zone_23_start=self.blend_zone_23_start,
            blend_zone_23_end=self.blend_zone_23_end,
            blend_zone_34_start=self.blend_zone_34_start,
            blend_zone_34_end=self.blend_zone_34_end
        )

    def set_parameters(self, params: StitchingParameters4):
        """Установить параметры из загруженного объекта"""
        for key, value in asdict(params).items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._blend_masks_computed = True

    def calibrate(self, calibration_file: str) -> None:
        """Выполнить калибровку и сохранить параметры"""
        self.initialize_stitching_parameters()
        
        # Захватываем первые кадры для калибровки обработки
        cap1 = cv2.VideoCapture(self.video1_path)
        cap2 = cv2.VideoCapture(self.video2_path)
        cap3 = cv2.VideoCapture(self.video3_path)
        cap4 = cv2.VideoCapture(self.video4_path)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()
        
        if not ret1 or not ret2 or not ret3 or not ret4:
            raise Exception("Не удалось прочитать кадры для калибровки")
        
        first_stitched = self.stitch_frame_optimized(frame1, frame2, frame3, frame4)
        
        self.projection_map_x, self.projection_map_y = self.create_cylindrical_map(
            self.output_size[0], self.output_size[1])
        
        self.analyze_and_compute_crop_params(first_stitched)
        
        params = self.get_parameters()
        params.save(calibration_file)
        
        cap1.release()
        cap2.release()
        cap3.release()
        cap4.release()
        
        logger.info(f"Калибровка завершена. Параметры сохранены в {calibration_file}")

    def process_with_params(self, frame1: np.ndarray, frame2: np.ndarray,
                            frame3: np.ndarray, frame4: np.ndarray) -> np.ndarray:
        """Обработка четырех кадров с использованием загруженных параметров"""
        stitched = self.stitch_frame_optimized(frame1, frame2, frame3, frame4)
        processed = self.process_frame_full_pipeline(stitched)
        return processed
