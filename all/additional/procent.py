import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
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

class Overlap_Finding:
    """
    Класс для нахождения перекрытия двух видео
    """
    
    def __init__(self, video1_path: str, video2_path: str):
        """
        Инициализация класса для сшивки видео
        
        Args:
            video1_path: Путь к первому видео
            video2_path: Путь ко второму видео
        """
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def calculate_overlap_percentage(self) -> Dict:
        """
        Расчет процента перекрытия двух видео на основе первого кадра
        
        Returns:
            Dict: информация о перекрытии
        """
        logger.info("=" * 50)
        logger.info("АНАЛИЗ ПЕРЕКРЫТИЯ ВИДЕО")
        logger.info("=" * 50)
        
        cap1 = cv2.VideoCapture(self.video1_path)
        cap2 = cv2.VideoCapture(self.video2_path)
        
        if not cap1.isOpened() or not cap2.isOpened():
            logger.error("Не удалось открыть видео для анализа перекрытия")
            return {}
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            logger.error("Не удалось прочитать первые кадры")
            cap1.release()
            cap2.release()
            return {}
        
        total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        
        # Анализируем перекрытие
        result = self._analyze_single_frame_overlap(frame1, frame2, 0)
        
        cap1.release()
        cap2.release()
        
        if result['overlap'] == 0:
            logger.warning("Не удалось обнаружить перекрытие между видео")
            self.overlap_results = {
                'overlap_percentage': 0,
                'matches': 0,
                'inliers': 0,
                'quality': 0,
                'homography': None,
                'overlap_area': 0,
                'total_area': 0,
                'video_info': {
                    'video1_frames': total_frames1,
                    'video2_frames': total_frames2,
                    'fps1': fps1,
                    'fps2': fps2,
                    'duration1': total_frames1 / fps1 if fps1 > 0 else 0,
                    'duration2': total_frames2 / fps2 if fps2 > 0 else 0
                }
            }
        else:            
            self.overlap_results = {
                'overlap_percentage': result['overlap'],
                'matches': result['matches'],
                'inliers': result['inliers'],
                'quality_score': result['quality'],
                'homography': result['homography'],
                'overlap_area_pixels': result['overlap_area'],
                'total_area_pixels': result['total_area'],
                'video_info': {
                    'video1_frames': total_frames1,
                    'video2_frames': total_frames2,
                    'fps1': fps1,
                    'fps2': fps2,
                    'duration1': total_frames1 / fps1 if fps1 > 0 else 0,
                    'duration2': total_frames2 / fps2 if fps2 > 0 else 0,
                    'size1': f"{frame1.shape[1]}x{frame1.shape[0]}",
                    'size2': f"{frame2.shape[1]}x{frame2.shape[0]}"
                }
            }
        
        self._log_overlap_results(self.overlap_results)
        
        return self.overlap_results
    
    def _analyze_single_frame_overlap(self, frame1: np.ndarray, frame2: np.ndarray, 
                                      frame_num: int) -> Dict:
        """
        Анализ перекрытия для одной пары кадров
        
        Args:
            frame1: первый кадр
            frame2: второй кадр
            frame_num: номер кадра
        
        Returns:
            Dict: информация о перекрытии
        """
        scale = 0.5
        h1, w1 = frame1.shape[:2]
        new_w = int(w1 * scale)
        new_h = int(h1 * scale)
        
        img1 = cv2.resize(frame1, (new_w, new_h))
        img2 = cv2.resize(frame2, (new_w, new_h))
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        kp1, desc1 = self.sift.detectAndCompute(gray1, None)
        kp2, desc2 = self.sift.detectAndCompute(gray2, None)
        
        if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
            logger.warning(f"Недостаточно признаков: {len(kp1) if kp1 else 0}, {len(kp2) if kp2 else 0}")
            return {
                'overlap': 0,
                'matches': 0,
                'inliers': 0,
                'quality': 0,
                'homography': None,
                'overlap_area': 0,
                'total_area': 0
            }
        
        matches = self.flann.knnMatch(desc1, desc2, k=2)
        
        # Применяем тест Лоу
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            logger.warning(f"Недостаточно хороших совпадений: {len(good_matches)}")
            return {
                'overlap': 0,
                'matches': len(good_matches),
                'inliers': 0,
                'quality': 0,
                'homography': None,
                'overlap_area': 0,
                'total_area': 0
            }
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            logger.warning("Не удалось найти гомографию")
            return {
                'overlap': 0,
                'matches': len(good_matches),
                'inliers': 0,
                'quality': 0,
                'homography': None,
                'overlap_area': 0,
                'total_area': 0
            }
        
        inliers = np.sum(mask)
        quality = inliers / len(good_matches) if len(good_matches) > 0 else 0
        
        h, w = img2.shape[:2]
        
        corners1 = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        corners1_transformed = cv2.perspectiveTransform(corners1, H)
        
        overlap_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(overlap_mask, [np.int32(corners1_transformed)], 255)
        
        overlap_area = np.sum(overlap_mask > 0)
        total_area = h * w
        overlap_percentage = (overlap_area / total_area) * 100
        
        
        if quality < 0.3:
            overlap_percentage *= quality * 3  
        
        return {
            'overlap': overlap_percentage,
            'matches': len(good_matches),
            'inliers': int(inliers),
            'quality': float(quality),
            'homography': H,
            'overlap_area': int(overlap_area),
            'total_area': total_area
        }   
    
    def _log_overlap_results(self, results: Dict) -> None:
        """
        Логирование результатов анализа перекрытия
        
        Args:
            results: результаты анализа
        """
        if not results:
            return
        
        logger.info("=" * 50)
        logger.info("РЕЗУЛЬТАТЫ АНАЛИЗА ПЕРЕКРЫТИЯ")
        logger.info("=" * 50)
        
        if results.get('overlap_percentage', 0) == 0:
            logger.warning("Перекрытие не обнаружено!")
            return
        
        logger.info(f"Процент перекрытия: {results['overlap_percentage']:.1f}%")
        logger.info(f"Качество: {results.get('quality_label', 'N/A')}")
        logger.info(f"Количество совпадений: {results.get('matches', 0)}")
        logger.info(f"Количество inliers: {results.get('inliers', 0)}")
        logger.info(f"Качество совмещения: {results.get('quality_score', 0):.2f}")
        logger.info(f"Площадь перекрытия: {results.get('overlap_area_pixels', 0)} пикселей")
        
        if 'video_info' in results:
            info = results['video_info']
            logger.info(f"Размер видео 1: {info.get('size1', 'N/A')}")
            logger.info(f"Размер видео 2: {info.get('size2', 'N/A')}")
        
        logger.info(f"Рекомендации: {results.get('recommendations', 'N/A')}")
        logger.info("=" * 50)
    
    def visualize_overlap(self, output_path: str = None) -> np.ndarray:
        """
        Визуализация перекрытия на первом кадре
        
        Args:
            output_path: путь для сохранения визуализации (опционально)
            
        Returns:
            np.ndarray: изображение с визуализацией
        """
        if self.overlap_results is None:
            logger.warning("Сначала выполните calculate_overlap_percentage()")
            return None
        
        cap1 = cv2.VideoCapture(self.video1_path)
        cap2 = cv2.VideoCapture(self.video2_path)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        cap1.release()
        cap2.release()
        
        if not ret1 or not ret2:
            logger.error("Не удалось прочитать кадры для визуализации")
            return None
        
        scale = 0.5
        h1, w1 = frame1.shape[:2]
        new_w = int(w1 * scale)
        new_h = int(h1 * scale)
        
        img1 = cv2.resize(frame1, (new_w, new_h))
        img2 = cv2.resize(frame2, (new_w, new_h))
        
        if self.overlap_results['homography'] is not None:
            H = self.overlap_results['homography']
            
            h, w = img2.shape[:2]
            corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, H)
            
            vis_img = img2.copy()
            cv2.polylines(vis_img, [np.int32(transformed_corners)], True, (0, 255, 0), 3)
            
            overlap_mask = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.fillPoly(overlap_mask, [np.int32(transformed_corners)], (0, 255, 0))
            vis_img = cv2.addWeighted(vis_img, 1, overlap_mask, 0.3, 0)
            
            info_text = [
                f"Перекрытие: {self.overlap_results['overlap_percentage']:.1f}%",
                f"Качество: {self.overlap_results.get('quality_label', 'N/A')}",
                f"Совпадений: {self.overlap_results['matches']}",
                f"Inliers: {self.overlap_results['inliers']}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(vis_img, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_img, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                y_offset += 30
            
            cv2.imshow('Overlap Visualization', vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if output_path:
                cv2.imwrite(output_path, vis_img)
                logger.info(f"Визуализация сохранена в {output_path}")
            
            return vis_img
        else:
            logger.warning("Нет гомографии для визуализации")
            return None

    

if __name__ == "__main__":
    stitcher = Overlap_Finding(
        video1_path="Video2.avi",
        video2_path="Video3_resized.avi")
    
    overlap_results = stitcher.calculate_overlap_percentage()
    stitcher.visualize_overlap("overlap_visualization.jpg")
