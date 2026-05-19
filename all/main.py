import argparse
import cv2
import numpy as np
import yaml
import os
import time
from video_proc import Detection
from stitch2 import OptimizedCylindricalStitcher2, StitchingParameters2
from stitch3 import OptimizedCylindricalStitcher3, StitchingParameters3
from stitch4 import OptimizedCylindricalStitcher4, StitchingParameters4
from stitch5 import OptimizedCylindricalStitcher5, StitchingParameters5
from stitch6 import OptimizedCylindricalStitcher6, StitchingParameters6


class CameraCalibration:
    """Класс для удаления дисторсии"""
    
    def __init__(self, calibration_path: str = 'calibration_results.npz'):
        if not os.path.exists(calibration_path):
            raise FileNotFoundError(f"Файл калибровки не найден: {calibration_path}")
        
        calibration_data = np.load(calibration_path)
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coeffs = calibration_data['dist_coeffs']
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted


class RealTimeVideoProcessor:
    """Основной класс для обработки 1, 2, 3, 4, 5 или 6 видеопотоков"""
    
    def __init__(self, config_path: str, calibration_path: str = 'calibration_results.npz',
                 mode: str = 'process', calibration_file: str = 'stitching_params.pkl'):
        with open(config_path, 'r', encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        
        # Определяем количество камер из конфига
        self.rtsp_urls = []
        
        # Поддержка формата rtsp_url как списка
        if 'rtsp_url' in self.config:
            urls = self.config['rtsp_url']
            if isinstance(urls, list):
                self.rtsp_urls = urls
            elif isinstance(urls, str):
                self.rtsp_urls = [urls]
        
        # Поддержка формата rtsp_url_1, rtsp_url_2, rtsp_url_3, rtsp_url_4, rtsp_url_5, rtsp_url_6
        for i in range(1, 7):
            key = f'rtsp_url_{i}'
            if key in self.config:
                if len(self.rtsp_urls) < i:
                    self.rtsp_urls.append(self.config[key])
        
        self.num_cameras = len(self.rtsp_urls)
        
        if self.num_cameras == 0:
            raise ValueError("Не указаны URL видеопотоков в конфигурации. "
                           "Используйте rtsp_url (список) или rtsp_url_1, rtsp_url_2, ... rtsp_url_6")
        
        if self.num_cameras > 6:
            print(f"Предупреждение: обнаружено {self.num_cameras} камер, поддерживается максимум 6")
            self.rtsp_urls = self.rtsp_urls[:6]
            self.num_cameras = 6
        
        required_params = ['model_path', 'confidence', 'skip_frames', 'scale', 'save_path']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Отсутствует обязательный параметр конфигурации: {param}")
        
        self.skip_frames = self.config.get('skip_frames', 1)
        self.output_path = self.config.get('save_path', 'output.mp4')
        
        # Параметры для сшивки
        self.num_calibration_frames = self.config.get('num_calibration_frames', 10)
        self.neutral_plane_t = self.config.get('neutral_plane_t', 0.5)
        self.fov_horizontal = self.config.get('fov_horizontal', 150)
        self.adaptive_smoothness = self.config.get('adaptive_smoothness', 50.0)
        self.crop_percent = self.config.get('crop_percent', 0.15)
        
        self.mode = mode
        self.calibration_file = calibration_file
        
        print(f"Количество камер: {self.num_cameras}")
        print(f"Режим работы: {'калибровка' if mode == 'calibrate' else 'обработка'}")
        
        if self.num_cameras > 1:
            if self.num_cameras == 2:
                print(f"Параметры сшивки (2 камеры): кадров для калибровки={self.num_calibration_frames}, "
                      f"t={self.neutral_plane_t}, FOV={self.fov_horizontal}°, "
                      f"гладкость={self.adaptive_smoothness}, "
                      f"обрезка боков={self.crop_percent*100:.1f}%")
            elif self.num_cameras == 3:
                print(f"Параметры сшивки (3 камеры): кадров для калибровки={self.num_calibration_frames}, "
                      f"FOV={self.fov_horizontal}°, гладкость={self.adaptive_smoothness}, "
                      f"обрезка боков={self.crop_percent*100:.1f}%")
            elif self.num_cameras == 4:
                print(f"Параметры сшивки (4 камеры): кадров для калибровки={self.num_calibration_frames}, "
                      f"t={self.neutral_plane_t}, FOV={self.fov_horizontal}°, "
                      f"гладкость={self.adaptive_smoothness}, "
                      f"обрезка боков={self.crop_percent*100:.1f}%")
            elif self.num_cameras == 5:
                print(f"Параметры сшивки (5 камер): кадров для калибровки={self.num_calibration_frames}, "
                      f"t={self.neutral_plane_t}, FOV={self.fov_horizontal}°, "
                      f"гладкость={self.adaptive_smoothness}, "
                      f"обрезка боков={self.crop_percent*100:.1f}%")
            elif self.num_cameras == 6:
                print(f"Параметры сшивки (6 камер): кадров для калибровки={self.num_calibration_frames}, "
                      f"t={self.neutral_plane_t}, FOV={self.fov_horizontal}°, "
                      f"гладкость={self.adaptive_smoothness}, "
                      f"обрезка боков={self.crop_percent*100:.1f}%")
        
        self.calibration = CameraCalibration(calibration_path)
        self.detector = Detection(config_path)
        
        # Сшиватель будет создан позже в зависимости от количества камер
        self.stitcher = None
        self.stitcher_type = None  # '2', '3', '4', '5', '6'
        
        self.caps = []
        self.video_writer = None
        
        self.frame_counter = 0
        self.saved_frames = 0
        self.start_time = None
        
        self.output_width = None
        self.output_height = None
    
    def initialize_video_streams(self) -> bool:
        """Инициализация видеопотоков"""
        fps_values = []
        
        for i, url in enumerate(self.rtsp_urls):
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                print(f"Ошибка: Не удалось открыть видеопоток {i+1}: {url}")
                self._release_caps()
                return False
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_values.append(fps)
            
            ret, frame = cap.read()
            if not ret:
                print(f"Ошибка: Не удалось прочитать тестовый кадр из потока {i+1}")
                self._release_caps()
                return False
            
            print(f"Размер потока {i+1}: {frame.shape[1]}x{frame.shape[0]}")
            self.caps.append(cap)
        
        # Определяем FPS
        valid_fps = [f for f in fps_values if f > 0]
        self.fps = min(valid_fps) if valid_fps else 30.0
        print(f"Частота кадров: {self.fps} FPS")
        
        # Сбрасываем позицию
        for cap in self.caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return True
    
    def _release_caps(self):
        """Освобождение всех захватов"""
        for cap in self.caps:
            if cap and cap.isOpened():
                cap.release()
        self.caps.clear()
    
    def collect_calibration_frames(self):
        """Сбор кадров для калибровки сшивателя"""
        print(f"Сбор {self.num_calibration_frames} кадров для калибровки...")
        
        calibration_frames = [[] for _ in range(self.num_cameras)]
        
        for i in range(self.num_calibration_frames):
            frames = []
            all_ok = True
            
            for cap in self.caps:
                ret, frame = cap.read()
                if not ret:
                    all_ok = False
                    break
                frames.append(frame)
            
            if not all_ok:
                print(f"Предупреждение: собрано только {i} кадров из {self.num_calibration_frames}")
                break
            
            # Удаление дисторсии для всех кадров
            undistorted_frames = [self.calibration.undistort_image(f) for f in frames]
            
            for j, uf in enumerate(undistorted_frames):
                calibration_frames[j].append(uf)
            
            if (i + 1) % 5 == 0:
                print(f"  Собрано кадров: {i + 1}/{self.num_calibration_frames}")
        
        # Сбрасываем позицию
        for cap in self.caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        if len(calibration_frames[0]) < 5:
            print(f"Ошибка: недостаточно кадров для калибровки ({len(calibration_frames[0])} из 5)")
            return None
        
        print(f"Собрано {len(calibration_frames[0])} кадров для калибровки")
        return calibration_frames
    
    def save_calibration_videos(self, all_frames) -> list:
        """Сохранение калибровочных видео во временные файлы"""
        if not all_frames or len(all_frames[0]) < 5:
            return []
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = all_frames[0][0].shape[:2]
        
        print(f"Сохранение калибровочных видео размером {width}x{height}...")
        
        temp_files = []
        for i, frames in enumerate(all_frames):
            filename = f'calib_video_{i}_temp.mp4'
            out = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()
            temp_files.append(filename)
            print(f"  Сохранено: {filename} ({len(frames)} кадров)")
        
        return temp_files
    
    def run_calibration(self):
        """Запуск режима калибровки"""
        print(f"\n=== Запуск калибровки сшивки ({self.num_cameras} камер) ===")
        
        if not self.initialize_video_streams():
            print("Ошибка инициализации видеопотоков")
            return
        
        if self.num_cameras == 1:
            print("Одиночная камера не требует калибровки сшивки")
            self._release_caps()
            return
        
        calib_frames = self.collect_calibration_frames()
        
        if calib_frames is None:
            print("Ошибка сбора кадров для калибровки")
            self._release_caps()
            return
        
        temp_files = self.save_calibration_videos(calib_frames)
        
        if not temp_files:
            print("Ошибка сохранения калибровочных видео")
            self._release_caps()
            return
        
        print(f"\nИнициализация сшивателя для {self.num_cameras} камер...")
        
        try:
            if self.num_cameras == 2:
                # Используем OptimizedCylindricalStitcher2 из stitch2.py
                self.stitcher = OptimizedCylindricalStitcher2(
                    video1_path=temp_files[0],
                    video2_path=temp_files[1],
                    output_path='temp_output',
                    num_calibration_frames=len(calib_frames[0]),
                    neutral_plane_t=self.neutral_plane_t,
                    fov_horizontal=self.fov_horizontal,
                    adaptive_smoothness=self.adaptive_smoothness,
                    crop_percent=self.crop_percent
                )
                self.stitcher_type = '2'
                
            elif self.num_cameras == 3:
                # Используем OptimizedCylindricalStitcher3 из stitch3.py
                self.stitcher = OptimizedCylindricalStitcher3(
                    video_left_path=temp_files[0],
                    video_center_path=temp_files[1],
                    video_right_path=temp_files[2],
                    output_path='temp_output',
                    num_calibration_frames=len(calib_frames[0]),
                    fov_horizontal=self.fov_horizontal,
                    adaptive_smoothness=self.adaptive_smoothness,
                    crop_percent=self.crop_percent
                )
                self.stitcher_type = '3'
                
            elif self.num_cameras == 4:
                # Используем OptimizedCylindricalStitcher4 из stitch4.py
                self.stitcher = OptimizedCylindricalStitcher4(
                    video1_path=temp_files[0],
                    video2_path=temp_files[1],
                    video3_path=temp_files[2],
                    video4_path=temp_files[3],
                    output_path='temp_output',
                    num_calibration_frames=len(calib_frames[0]),
                    neutral_plane_t=self.neutral_plane_t,
                    fov_horizontal=self.fov_horizontal,
                    adaptive_smoothness=self.adaptive_smoothness,
                    crop_percent=self.crop_percent
                )
                self.stitcher_type = '4'
                
            elif self.num_cameras == 5:
                # Используем OptimizedCylindricalStitcher5 из stitch5.py
                self.stitcher = OptimizedCylindricalStitcher5(
                    video1_path=temp_files[0],
                    video2_path=temp_files[1],
                    video3_path=temp_files[2],
                    video4_path=temp_files[3],
                    video5_path=temp_files[4],
                    output_path='temp_output',
                    num_calibration_frames=len(calib_frames[0]),
                    neutral_plane_t=self.neutral_plane_t,
                    fov_horizontal=self.fov_horizontal,
                    adaptive_smoothness=self.adaptive_smoothness,
                    crop_percent=self.crop_percent
                )
                self.stitcher_type = '5'
                
            elif self.num_cameras == 6:
                # Используем OptimizedCylindricalStitcher6 из stitch6.py
                self.stitcher = OptimizedCylindricalStitcher6(
                    video1_path=temp_files[0],
                    video2_path=temp_files[1],
                    video3_path=temp_files[2],
                    video4_path=temp_files[3],
                    video5_path=temp_files[4],
                    video6_path=temp_files[5],
                    output_path='temp_output',
                    num_calibration_frames=len(calib_frames[0]),
                    neutral_plane_t=self.neutral_plane_t,
                    fov_horizontal=self.fov_horizontal,
                    adaptive_smoothness=self.adaptive_smoothness,
                    crop_percent=self.crop_percent
                )
                self.stitcher_type = '6'
            
            # Выполняем калибровку и сохраняем параметры
            print("Выполнение калибровки...")
            self.stitcher.calibrate(self.calibration_file)
            
            print(f"\nКалибровка завершена успешно!")
            print(f"Параметры сохранены в: {self.calibration_file}")
            
        except Exception as e:
            print(f"Ошибка при калибровке: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Удаляем временные файлы
            print("Очистка временных файлов...")
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
            
            self._release_caps()
    
    def initialize_stitcher_from_file(self) -> bool:
        """Инициализация сшивателя из файла параметров"""
        if not os.path.exists(self.calibration_file):
            print(f"Ошибка: Файл параметров не найден: {self.calibration_file}")
            print("Сначала выполните калибровку с параметром --mode calibrate")
            return False
        
        print(f"Загрузка параметров из {self.calibration_file}...")
        
        try:
            # Пробуем загрузить как параметры для 2 камер
            try:
                params2 = StitchingParameters2.load(self.calibration_file)
                self.stitcher = OptimizedCylindricalStitcher2(
                    fov_horizontal=params2.fov_horizontal,
                    adaptive_smoothness=params2.adaptive_smoothness,
                    crop_percent=params2.crop_percent
                )
                self.stitcher.set_parameters(params2)
                self.stitcher_type = '2'
                print(f"Загружены параметры для 2 камер")
                return True
            except:
                pass
            
            # Пробуем загрузить как параметры для 3 камер
            try:
                params3 = StitchingParameters3.load(self.calibration_file)
                self.stitcher = OptimizedCylindricalStitcher3(
                    fov_horizontal=params3.fov_horizontal,
                    adaptive_smoothness=params3.adaptive_smoothness,
                    crop_percent=params3.crop_percent
                )
                self.stitcher.set_parameters(params3)
                self.stitcher_type = '3'
                print(f"Загружены параметры для 3 камер")
                return True
            except:
                pass
            
            # Пробуем загрузить как параметры для 4 камер
            try:
                params4 = StitchingParameters4.load(self.calibration_file)
                self.stitcher = OptimizedCylindricalStitcher4(
                    fov_horizontal=params4.fov_horizontal,
                    adaptive_smoothness=params4.adaptive_smoothness,
                    crop_percent=params4.crop_percent
                )
                self.stitcher.set_parameters(params4)
                self.stitcher_type = '4'
                print(f"Загружены параметры для 4 камер")
                return True
            except:
                pass
            
            # Пробуем загрузить как параметры для 5 камер
            try:
                params5 = StitchingParameters5.load(self.calibration_file)
                self.stitcher = OptimizedCylindricalStitcher5(
                    fov_horizontal=params5.fov_horizontal,
                    adaptive_smoothness=params5.adaptive_smoothness,
                    crop_percent=params5.crop_percent
                )
                self.stitcher.set_parameters(params5)
                self.stitcher_type = '5'
                print(f"Загружены параметры для 5 камер")
                return True
            except:
                pass
            
            # Пробуем загрузить как параметры для 6 камер
            try:
                params6 = StitchingParameters6.load(self.calibration_file)
                self.stitcher = OptimizedCylindricalStitcher6(
                    fov_horizontal=params6.fov_horizontal,
                    adaptive_smoothness=params6.adaptive_smoothness,
                    crop_percent=params6.crop_percent
                )
                self.stitcher.set_parameters(params6)
                self.stitcher_type = '6'
                print(f"Загружены параметры для 6 камер")
                return True
            except:
                pass
            
            print(f"Ошибка: Не удалось загрузить параметры для известного типа сшивателя")
            return False
            
        except Exception as e:
            print(f"Ошибка при загрузке параметров: {e}")
            return False
    
    def initialize_video_writer(self) -> bool:
        """Инициализация VideoWriter"""
        if self.num_cameras == 1:
            # Для одиночной камеры используем размер кадра
            ret, frame = self.caps[0].read()
            self.caps[0].set(cv2.CAP_PROP_POS_FRAMES, 0)
            if ret:
                self.output_width = frame.shape[1]
                self.output_height = frame.shape[0]
            else:
                self.output_width = 1920
                self.output_height = 1080
        else:
            self.output_width = self.stitcher.final_output_size[0]
            self.output_height = self.stitcher.final_output_size[1]
        
        # Убеждаемся, что размеры четные
        if self.output_width % 2 != 0:
            self.output_width -= 1
        if self.output_height % 2 != 0:
            self.output_height -= 1
        
        print(f"Инициализация VideoWriter для файла: {self.output_path}")
        print(f"Размер видео: {self.output_width}x{self.output_height}, FPS: {self.fps}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.fps, 
            (self.output_width, self.output_height)
        )
        
        if not self.video_writer.isOpened():
            print("Ошибка: Не удалось открыть VideoWriter")
            return False
        
        return True
    
    def process_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """Обработка одиночного кадра (без сшивки)"""
        # Удаление дисторсии
        frame_undistorted = self.calibration.undistort_image(frame)
        
        # Детекция людей
        people = self.detector.process_frame(frame_undistorted)
        result_frame = self.detector.drawing(frame_undistorted, people)
        
        return result_frame
    
    def process_multi_frame(self, frames: list) -> np.ndarray:
        """Обработка нескольких кадров со сшивкой"""
        # Удаление дисторсии
        undistorted_frames = [self.calibration.undistort_image(f) for f in frames]
        
        # Сшивка в зависимости от типа сшивателя
        if self.stitcher_type == '2':
            processed_frame = self.stitcher.process_with_params(
                undistorted_frames[0], undistorted_frames[1])
        elif self.stitcher_type == '3':
            processed_frame = self.stitcher.process_with_params(
                undistorted_frames[0], undistorted_frames[1], undistorted_frames[2])
        elif self.stitcher_type == '4':
            processed_frame = self.stitcher.process_with_params(
                undistorted_frames[0], undistorted_frames[1], 
                undistorted_frames[2], undistorted_frames[3])
        elif self.stitcher_type == '5':
            processed_frame = self.stitcher.process_with_params(
                undistorted_frames[0], undistorted_frames[1], 
                undistorted_frames[2], undistorted_frames[3], undistorted_frames[4])
        elif self.stitcher_type == '6':
            processed_frame = self.stitcher.process_with_params(
                undistorted_frames[0], undistorted_frames[1], 
                undistorted_frames[2], undistorted_frames[3], 
                undistorted_frames[4], undistorted_frames[5])
        else:
            return undistorted_frames[0]
        
        # Детекция людей
        people = self.detector.process_frame(processed_frame)
        result_frame = self.detector.drawing(processed_frame, people)
        
        return result_frame
    
    def run_processing(self):
        """Основной цикл обработки"""
        print(f"\n=== Запуск обработки видеопотоков ({self.num_cameras} камер) ===")
        print("Нажмите 'q' для выхода")
        print("Нажмите 'p' для паузы/продолжения")
        print("Нажмите 's' для сохранения скриншота")
        
        if not self.initialize_video_streams():
            print("Ошибка инициализации видеопотоков")
            return
        
        if self.num_cameras > 1:
            if not self.initialize_stitcher_from_file():
                print("Ошибка инициализации сшивателя")
                self._release_caps()
                return
        elif self.num_cameras != len(self.caps):
            print(f"Предупреждение: количество камер в конфиге ({self.num_cameras}) "
                  f"не соответствует открытым потокам ({len(self.caps)})")
        
        if not self.initialize_video_writer():
            print("Ошибка инициализации VideoWriter")
            self._release_caps()
            return
        
        # Настройка окна
        if self.num_cameras == 1:
            window_name = "Видео с детекцией людей"
        else:
            window_name = f"Сшитое видео с детекцией людей ({self.num_cameras} камеры)"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.output_width // 2, self.output_height // 2)
        
        self.start_time = time.time()
        last_stat_time = self.start_time
        is_paused = False
        
        try:
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Получена команда выхода")
                    break
                elif key == ord('p'):
                    is_paused = not is_paused
                    status = "включена" if is_paused else "выключена"
                    print(f"Пауза: {status}")
                elif key == ord('s'):
                    # Сохранение скриншота
                    if hasattr(self, 'last_result_frame'):
                        screenshot_path = f"screenshot_{self.saved_frames:06d}.jpg"
                        cv2.imwrite(screenshot_path, self.last_result_frame)
                        print(f"Сохранен скриншот: {screenshot_path}")
                
                if is_paused:
                    time.sleep(0.1)
                    continue
                
                # Читаем кадры со всех камер
                frames = []
                all_ok = True
                for cap in self.caps:
                    ret, frame = cap.read()
                    if not ret:
                        all_ok = False
                        break
                    frames.append(frame)
                
                if not all_ok:
                    print("Достигнут конец видео или ошибка чтения")
                    break
                
                self.frame_counter += 1
                
                # Пропуск кадров
                if self.skip_frames > 0 and self.frame_counter % (self.skip_frames + 1) != 0:
                    if hasattr(self, 'last_result_frame'):
                        cv2.imshow(window_name, self.last_result_frame)
                    continue
                
                # Обрабатываем в зависимости от количества камер
                if self.num_cameras == 1:
                    result_frame = self.process_single_frame(frames[0])
                else:
                    result_frame = self.process_multi_frame(frames)
                
                # Сохраняем и отображаем
                self.last_result_frame = result_frame
                cv2.imshow(window_name, result_frame)
                self.video_writer.write(result_frame)
                self.saved_frames += 1
                
                # Вывод статистики каждые 5 секунд
                current_time = time.time()
                if current_time - last_stat_time >= 5.0:
                    elapsed = current_time - self.start_time
                    fps = self.saved_frames / elapsed if elapsed > 0 else 0
                    print(f"Обработано: {self.saved_frames} кадров, "
                          f"всего прочитано: {self.frame_counter}, "
                          f"FPS: {fps:.1f}")
                    last_stat_time = current_time
                
        except KeyboardInterrupt:
            print("\nОбработка прервана пользователем")
        except Exception as e:
            print(f"Ошибка в основном цикле: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def run(self):
        """Запуск в зависимости от режима"""
        if self.mode == 'calibrate':
            self.run_calibration()
        else:
            self.run_processing()
    
    def cleanup(self):
        """Очистка ресурсов"""
        print("\nОчистка ресурсов...")
        
        for i, cap in enumerate(self.caps):
            if cap and cap.isOpened():
                cap.release()
                print(f"Видеопоток {i+1} освобожден")
        
        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
            print(f"VideoWriter закрыт. Видео сохранено в: {self.output_path}")
            
            if os.path.exists(self.output_path):
                file_size = os.path.getsize(self.output_path) / (1024 * 1024)
                print(f"Размер выходного файла: {file_size:.2f} MB")
        
        cv2.destroyAllWindows()
        
        if self.start_time and self.mode != 'calibrate':
            total_time = time.time() - self.start_time
            avg_fps = self.saved_frames / total_time if total_time > 0 else 0
            
            print("\n=== Финальная статистика ===")
            print(f"Количество камер: {self.num_cameras}")
            print(f"Всего кадров в потоке: {self.frame_counter}")
            print(f"Сохранено кадров: {self.saved_frames}")
            print(f"Пропущено кадров: {self.frame_counter - self.saved_frames}")
            print(f"Общее время: {total_time:.1f} секунд")
            print(f"Средний FPS: {avg_fps:.1f}")
            print(f"Размер видео: {self.output_width}x{self.output_height}")
        
        print("Обработка завершена")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Обработка видеопотоков (1-6 камеры)')
    parser.add_argument('--config', default='config.yaml', help='Конфигурационный файл')
    parser.add_argument('--calibration', default='calibration_results.npz', 
                       help='Файл калибровки камеры')
    parser.add_argument('--output', help='Выходной файл')
    parser.add_argument('--mode', choices=['calibrate', 'process'], default='process',
                       help='Режим работы: calibrate - калибровка, process - обработка')
    parser.add_argument('--params_file', default='stitching_params.pkl',
                       help='Файл для сохранения/загрузки параметров сшивки')
    
    args = parser.parse_args()
    
    try:
        processor = RealTimeVideoProcessor(
            args.config, 
            args.calibration,
            mode=args.mode,
            calibration_file=args.params_file
        )
        
        if args.output:
            processor.output_path = args.output
            
        processor.run()
    except ValueError as e:
        print(f"Ошибка конфигурации: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"Ошибка файла: {e}")
        return 1
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
