import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import os
import math
import logging
import sys
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class OptimizedCylindricalStitcher:
    """
    Оптимизированный статический сшиватель видео с GPU ускорением.
    Вычисляет все параметры один раз, затем только применяет трансформации.
    """
    
    def __init__(self, video1_path: str, video2_path: str, output_path: str,
                 num_calibration_frames: int = 10, neutral_plane_t: float = 0.5,
                 fov_horizontal: float = 150):
        """
        Инициализация статического сшивателя с GPU ускорением.
        
        Args:
            video1_path: Путь к первому видео
            video2_path: Путь ко второму видео
            output_path: Путь для сохранения результата
            num_calibration_frames: Количество кадров для калибровки
            neutral_plane_t: Параметр нейтральной плоскости
            fov_horizontal: Горизонтальный угол обзора
        """
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.output_path = output_path
        self.num_calibration_frames = num_calibration_frames
        self.neutral_plane_t = neutral_plane_t
        self.fov_horizontal = fov_horizontal
        
        if not 0 <= neutral_plane_t <= 1:
            raise ValueError("neutral_plane_t должен быть в диапазоне от 0 до 1")
        
        self.device = self._init_torch_device()
        
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
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
        
        self.cylindrical_map_x = None
        self.cylindrical_map_y = None
        self.horizontal_crop_slices = None
        self.final_crop_slices = None
        self.final_output_size = None
        self.horizontal_crop_percent = 0.30
        
        self.warp_grid_1 = None
        self.warp_grid_2 = None
        self.remap_grid = None
        self.blend_mask_tensor = None
        self.crop_tensor = None
        self.crop_offset_x = 0
        self.crop_offset_y = 0
        self.initialized = False
    
    def _init_torch_device(self):
        """Инициализация PyTorch устройства."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Используется GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("Используется CPU")
        return device
    
    def _numpy_to_torch(self, image: np.ndarray) -> torch.Tensor:
        """Конвертация numpy изображения в PyTorch tensor."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().to(self.device) / 255.0
        return tensor
    
    def _torch_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Конвертация PyTorch tensor в numpy изображение."""
        image = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    
    def _create_warp_grid(self, M: np.ndarray, src_size: Tuple[int, int], 
                         dst_size: Tuple[int, int]) -> torch.Tensor:
        """Создание warp grid для быстрой трансформации."""
        h_src, w_src = src_size
        w_dst, h_dst = dst_size
        
        y_dst, x_dst = torch.meshgrid(
            torch.linspace(0, h_dst - 1, h_dst),
            torch.linspace(0, w_dst - 1, w_dst),
            indexing='ij'
        )
        
        ones = torch.ones_like(x_dst)
        coords_dst = torch.stack([x_dst, y_dst, ones], dim=-1).float().to(self.device)
        
        M_tensor = torch.from_numpy(M).float().to(self.device)
        M_inv = torch.linalg.inv(M_tensor)
        
        coords_src = torch.matmul(coords_dst.reshape(-1, 3), M_inv.T)
        coords_src = coords_src.reshape(h_dst, w_dst, 3)
        
        x_src = coords_src[..., 0] / coords_src[..., 2]
        y_src = coords_src[..., 1] / coords_src[..., 2]
        
        x_src_norm = (x_src / (w_src - 1)) * 2 - 1
        y_src_norm = (y_src / (h_src - 1)) * 2 - 1
        
        grid = torch.stack([x_src_norm, y_src_norm], dim=-1).unsqueeze(0)
        return grid
    
    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Извлечение признаков SIFT из изображения."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def find_homography(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        """Поиск матрицы гомографии между двумя изображениями."""
        kp1, desc1 = self.extract_features(img1)
        kp2, desc2 = self.extract_features(img2)
        
        if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
            return None
        
        matches = self.flann.knnMatch(desc1, desc2, k=2)
        good_matches = []
        
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    
    def calculate_homography_from_frames(self) -> None:
        """Вычисление матриц гомографии на основе нескольких кадров."""
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
            raise Exception("Не удалось вычислить матрицы гомографии")
    
    def neutral_plane_transform(self) -> np.ndarray:
        """Создание преобразования для нейтральной плоскости."""
        H2_to_1 = self.homography_matrix_2_to_1 / self.homography_matrix_2_to_1[2, 2]
        H2_neutral = (np.eye(3) * (1 - self.neutral_plane_t) + H2_to_1 * self.neutral_plane_t)
        H2_neutral = H2_neutral / H2_neutral[2, 2]
        return H2_neutral
    
    def calculate_new_homography_to_neutral_plane(self, frame1: np.ndarray, frame2: np.ndarray) -> None:
        """Вычисление новой гомографии к нейтральной плоскости."""
        h2, w2 = frame2.shape[:2]
        
        frame2_tensor = self._numpy_to_torch(frame2)
        frame2_tensor = frame2_tensor.unsqueeze(0)
        
        grid = self._create_warp_grid(self.neutral_transform_2, (h2, w2), (h2 * 2, w2 * 2))
        warped2_neutral_tensor = F.grid_sample(frame2_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        warped2_neutral = self._torch_to_numpy(warped2_neutral_tensor.squeeze(0))
        
        gray_warped = cv2.cvtColor(warped2_neutral, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_warped, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
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
            raise Exception("Не удалось вычислить новую гомографию")
    
    def calculate_final_transforms(self, frame1: np.ndarray, frame2: np.ndarray) -> None:
        """Вычисление финальных трансформаций для сшивки."""
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        corners1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
        corners1_transformed = cv2.perspectiveTransform(corners1.reshape(-1, 1, 2), self.new_homography_1_to_2_neutral)
        
        corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
        corners2_transformed = cv2.perspectiveTransform(corners2.reshape(-1, 1, 2), self.neutral_transform_2_corrected)
        
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
    
    def _create_blend_mask_torch(self) -> None:
        """Создание маски смешивания в виде PyTorch тензора."""
        h, w = self.output_size[1], self.output_size[0]
        
        test_frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        test_frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        
        warped1 = self._apply_warp(test_frame1, self.final_transform_1, self.output_size)
        warped2 = self._apply_warp(test_frame2, self.final_transform_2, self.output_size)
        
        mask1 = (warped1.sum(axis=2) > 10)
        mask2 = (warped2.sum(axis=2) > 10)
        overlap = mask1 & mask2
        
        if not np.any(overlap):
            blend_start = w // 2 - 150
            blend_end = w // 2 + 150
        else:
            overlap_cols = np.where(np.any(overlap, axis=0))[0]
            if len(overlap_cols) == 0:
                blend_start = w // 2 - 150
                blend_end = w // 2 + 150
            else:
                blend_start = overlap_cols[0]
                blend_end = overlap_cols[-1]
                
                blend_margin = 50
                blend_start = max(0, blend_start - blend_margin)
                blend_end = min(w, blend_end + blend_margin)
                
                min_blend_width = 200
                current_width = blend_end - blend_start
                if current_width < min_blend_width:
                    center = (blend_start + blend_end) // 2
                    blend_start = max(0, center - min_blend_width // 2)
                    blend_end = min(w, center + min_blend_width // 2)
        
        blend_mask = torch.zeros((h, w), dtype=torch.float32, device=self.device)
        
        blend_start = max(0, blend_start)
        blend_end = min(w, blend_end)
        overlap_width = blend_end - blend_start
        
        if overlap_width > 0:
            x_indices = torch.arange(blend_start, blend_end, device=self.device)
            t = (x_indices - blend_start) / overlap_width
            alpha = 1 / (1 + torch.exp(-12 * (t - 0.5)))
            blend_mask[:, blend_start:blend_end] = alpha.unsqueeze(0)
        
        self.blend_mask_tensor = blend_mask.unsqueeze(0).unsqueeze(0)
    
    def _apply_warp(self, image: np.ndarray, M: np.ndarray, size: tuple) -> np.ndarray:
        """Применение трансформации к изображению."""
        return cv2.warpPerspective(image, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    def create_cylindrical_map(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Создание LUT для цилиндрической проекции."""
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
        
        map_x = np.nan_to_num(map_x, nan=0.0, posinf=width-1, neginf=0.0)
        map_y = np.nan_to_num(map_y, nan=0.0, posinf=height-1, neginf=0.0)
        
        map_x = np.clip(map_x, 0, width - 1)
        map_y = np.clip(map_y, 0, height - 1)
        
        return map_x, map_y
    
    def _create_remap_grid(self) -> None:
        """Создание remap grid для цилиндрической проекции."""
        map_x_tensor = torch.from_numpy(self.cylindrical_map_x).float().to(self.device)
        map_y_tensor = torch.from_numpy(self.cylindrical_map_y).float().to(self.device)
        
        h_map, w_map = map_x_tensor.shape
        
        x_norm = (map_x_tensor / (w_map - 1)) * 2 - 1
        y_norm = (map_y_tensor / (h_map - 1)) * 2 - 1
        
        self.remap_grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
    
    def find_content_bounds(self, image: np.ndarray, black_threshold: int = 20) -> Tuple[int, int, int, int]:
        """Нахождение границ контента в изображении."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        height, width = gray.shape
        mask = gray > black_threshold
        
        if not np.any(mask):
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
    
    def _create_crop_tensor(self) -> None:
        """Создание тензора для обрезки."""
        test_frame = np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)
        
        cylindrical_frame = self._apply_cylindrical_projection(test_frame)
        
        crop_percent = self.horizontal_crop_percent
        crop_left = int(self.output_size[0] * crop_percent)
        crop_right = int(self.output_size[0] * crop_percent)
        
        x1 = crop_left
        y1 = 0
        x2 = self.output_size[0] - crop_right
        y2 = self.output_size[1]
        
        x1 = max(0, x1)
        x2 = min(self.output_size[0], x2)
        y1 = max(0, y1)
        y2 = min(self.output_size[1], y2)
        
        horizontally_cropped = cylindrical_frame[y1:y2, x1:x2]
        
        left, top, right, bottom = self.find_content_bounds(horizontally_cropped, black_threshold=15)
        
        min_width = 100
        min_height = 100
        
        if (right - left) < min_width:
            center_x = (left + right) // 2
            left = max(0, center_x - min_width // 2)
            right = min(horizontally_cropped.shape[1], left + min_width)
        
        if (bottom - top) < min_height:
            center_y = (top + bottom) // 2
            top = max(0, center_y - min_height // 2)
            bottom = min(horizontally_cropped.shape[0], top + min_height)
        
        self.crop_offset_x = x1 + left
        self.crop_offset_y = y1 + top
        crop_width = right - left
        crop_height = bottom - top
        
        crop_width = crop_width if crop_width % 2 == 0 else crop_width - 1
        crop_height = crop_height if crop_height % 2 == 0 else crop_height - 1
        
        crop_width = max(min_width, crop_width)
        crop_height = max(min_height, crop_height)
        
        if crop_width % 2 != 0:
            crop_width += 1
        if crop_height % 2 != 0:
            crop_height += 1
        
        self.final_output_size = (crop_width, crop_height)
        
        logger.info(f"Финальный размер (четный): {crop_width}x{crop_height}")
        
        self.crop_tensor = torch.tensor([
            [self.crop_offset_y, self.crop_offset_y + crop_height],
            [self.crop_offset_x, self.crop_offset_x + crop_width]
        ], device=self.device)
    
    def _apply_cylindrical_projection(self, frame: np.ndarray) -> np.ndarray:
        """Применение цилиндрической проекции."""
        if self.remap_grid is None:
            raise ValueError("Remap grid не инициализирован")
        
        frame_tensor = self._numpy_to_torch(frame).unsqueeze(0)
        result = F.grid_sample(frame_tensor, self.remap_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return self._torch_to_numpy(result.squeeze(0))
    
    def initialize_stitching(self) -> None:
        """Инициализация всех параметров сшивки один раз."""
        if self.initialized:
            return
        
        logger.info("Инициализация параметров сшивки...")
        
        self.calculate_homography_from_frames()
        
        cap1 = cv2.VideoCapture(self.video1_path)
        cap2 = cv2.VideoCapture(self.video2_path)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            raise Exception("Не удалось прочитать кадры для инициализации")
        
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        self.neutral_transform_2 = self.neutral_plane_transform()
        self.calculate_new_homography_to_neutral_plane(frame1, frame2)
        self.calculate_final_transforms(frame1, frame2)
        
        self.warp_grid_1 = self._create_warp_grid(self.final_transform_1, (h1, w1), self.output_size)
        self.warp_grid_2 = self._create_warp_grid(self.final_transform_2, (h2, w2), self.output_size)
        
        self._create_blend_mask_torch()
        
        self.cylindrical_map_x, self.cylindrical_map_y = self.create_cylindrical_map(
            self.output_size[0], self.output_size[1]
        )
        
        self._create_remap_grid()
        
        test_stitched = self._stitch_single_frame(frame1, frame2)
        self._create_crop_tensor()
        
        cap1.release()
        cap2.release()
        
        self.initialized = True
        logger.info("Параметры сшивки инициализированы!")
    
    def _stitch_single_frame(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Сшивка одного кадра с использованием предварительно вычисленных параметров."""
        frame1_tensor = self._numpy_to_torch(frame1).unsqueeze(0)
        frame2_tensor = self._numpy_to_torch(frame2).unsqueeze(0)
        
        warped1 = F.grid_sample(frame1_tensor, self.warp_grid_1, mode='bilinear', padding_mode='zeros', align_corners=True)
        warped2 = F.grid_sample(frame2_tensor, self.warp_grid_2, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        mask1 = (warped1.sum(dim=1, keepdim=True) > 0.01)
        mask2 = (warped2.sum(dim=1, keepdim=True) > 0.01)
        
        result = torch.zeros_like(warped1)
        
        result[mask1] = warped1[mask1]
        
        video2_only = mask2 & ~mask1
        result[video2_only] = warped2[video2_only]
        
        overlap = mask1 & mask2
        if overlap.any():
            blend_mask_3d = self.blend_mask_tensor.repeat(1, 3, 1, 1)
            blended = warped1 * (1 - blend_mask_3d) + warped2 * blend_mask_3d
            result[overlap] = blended[overlap]
        
        return self._torch_to_numpy(result.squeeze(0))
    
    def process_video(self) -> str:
        """Обработка всего видео с использованием предварительно вычисленных параметров."""
        if not self.initialized:
            self.initialize_stitching()
        
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
        
        logger.info(f"Начинаю обработку {total_frames} кадров...")
        logger.info(f"Финальный размер: {self.final_output_size}")
        
        output_path_with_ext = self.output_path
        if not output_path_with_ext.lower().endswith(('.mp4', '.avi')):
            output_path_with_ext = self.output_path + '.avi'
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path_with_ext, fourcc, fps, self.final_output_size)
        
        if not out.isOpened():
            raise Exception(f"Не удалось создать VideoWriter для размера {self.final_output_size}")
        
        frame_count = 0
        import time
        start_time = time.time()
        process_times = []
        
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            process_start = time.time()
            
            try:
                stitched = self._stitch_single_frame(frame1, frame2)
                cylindrical = self._apply_cylindrical_projection(stitched)
                
                y_start, y_end = int(self.crop_tensor[0, 0].item()), int(self.crop_tensor[0, 1].item())
                x_start, x_end = int(self.crop_tensor[1, 0].item()), int(self.crop_tensor[1, 1].item())
                
                cropped = cylindrical[y_start:y_end, x_start:x_end]
                
                final_frame = cropped
                if cropped.shape[:2] != self.final_output_size[::-1]:
                    final_frame = cv2.resize(cropped, self.final_output_size)
                
                out.write(final_frame)
                
                process_time = time.time() - process_start
                process_times.append(process_time)
                frame_count += 1
                
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    progress = (frame_count / total_frames) * 100
                    
                    avg_process = np.mean(process_times[-100:]) * 1000 if process_times else 0
                    
                    logger.info(f"Обработано: {frame_count}/{total_frames} ({progress:.1f}%), "
                              f"Скорость: {fps_actual:.1f} FPS, "
                              f"Обработка: {avg_process:.1f}ms")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке кадра {frame_count}: {e}")
                continue
        
        logger.info("Завершение обработки...")
        cap1.release()
        cap2.release()
        out.release()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        logger.info(f"Всего кадров: {frame_count}")
        logger.info(f"Общее время: {total_time:.1f} секунд")
        logger.info(f"Средняя скорость: {avg_fps:.1f} FPS")
        logger.info(f"Финальный размер: {self.final_output_size}")
        logger.info(f"Финальный файл: {output_path_with_ext}")
        
        return output_path_with_ext


