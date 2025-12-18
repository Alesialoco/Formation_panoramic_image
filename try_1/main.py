import argparse
import cv2
import numpy as np
import yaml
import os
import time
from video_proc import Detection
from stich import OptimizedCylindricalStitcher


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
    """Основной класс для обработки видеопотоков"""
    
    def __init__(self, config_path: str, calibration_path: str = 'calibration_results.npz'):
        with open(config_path, 'r', encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        
        required_params = ['rtsp_url_1', 'rtsp_url_2', 'model_path', 'confidence', 
                          'skip_frames', 'scale', 'save_path']
        
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Отсутствует обязательный параметр конфигурации: {param}")
        
        self.video1_url = self.config['rtsp_url_1']
        self.video2_url = self.config['rtsp_url_2']
        self.skip_frames = self.config.get('skip_frames', 1)
        self.output_path = self.config.get('save_path', 'output.mp4')
        
        self.num_calibration_frames = self.config.get('num_calibration_frames', 10)
        self.neutral_plane_t = self.config.get('neutral_plane_t', 0.5)
        self.fov_horizontal = self.config.get('fov_horizontal', 150)
        
        self.use_gpu = self.config.get('use_gpu', True)
        
        print(f"Параметры сшивки: кадров для калибровки={self.num_calibration_frames}, "
              f"t={self.neutral_plane_t}, FOV={self.fov_horizontal}°, GPU={self.use_gpu}")
        
        self.calibration = CameraCalibration(calibration_path)
        self.detector = Detection(config_path)
        self.stitcher = None
        
        self.cap1 = None
        self.cap2 = None
        self.video_writer = None
        
        self.frame_counter = 0
        self.saved_frames = 0
        self.start_time = None
    
    def initialize_video_streams(self) -> bool:
        """Инициализация видеопотоков"""
        self.cap1 = cv2.VideoCapture(self.video1_url)
        
        self.cap2 = cv2.VideoCapture(self.video2_url)
        
        if not self.cap1.isOpened():
            print(f"Ошибка: Не удалось открыть видеопоток 1")
            return False
        
        if not self.cap2.isOpened():
            print(f"Ошибка: Не удалось открыть видеопоток 2")
            if self.cap1.isOpened():
                self.cap1.release()
            return False
        
        self.fps1 = self.cap1.get(cv2.CAP_PROP_FPS)
        self.fps2 = self.cap2.get(cv2.CAP_PROP_FPS)
        self.fps = min(self.fps1, self.fps2)
        
        if self.fps <= 0:
            self.fps = 30.0
            print(f"Не удалось определить FPS, использую {self.fps}")
        else:
            print(f"Частота кадров: {self.fps} FPS")
        
        ret1, test_frame1 = self.cap1.read()
        ret2, test_frame2 = self.cap2.read()
        
        if not ret1 or not ret2:
            print("Ошибка: Не удалось прочитать тестовые кадры")
            return False
        
        self.width1, self.height1 = test_frame1.shape[1], test_frame1.shape[0]
        self.width2, self.height2 = test_frame2.shape[1], test_frame2.shape[0]
        
        print(f"Размер потока 1: {self.width1}x{self.height1}")
        print(f"Размер потока 2: {self.width2}x{self.height2}")
        
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return True
    
    def collect_calibration_frames(self):
        """Сбор кадров для калибровки сшивателя"""
        print(f"Сбор {self.num_calibration_frames} кадров для калибровки...")
        
        calib_frames1 = []
        calib_frames2 = []
        
        for i in range(self.num_calibration_frames):
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            
            if not ret1 or not ret2:
                print(f"Предупреждение: собрано только {i} кадров из {self.num_calibration_frames}")
                break
            
            frame1_undistorted = self.calibration.undistort_image(frame1)
            frame2_undistorted = self.calibration.undistort_image(frame2)
            
            calib_frames1.append(frame1_undistorted)
            calib_frames2.append(frame2_undistorted)
            
            if (i + 1) % 5 == 0:
                print(f"  Собрано кадров: {i + 1}/{self.num_calibration_frames}")
        
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        if len(calib_frames1) < 5:
            print(f"Ошибка: недостаточно кадров для калибровки ({len(calib_frames1)} из 5)")
            return None, None
        
        print(f"Собрано {len(calib_frames1)} кадров для калибровки")
        return calib_frames1, calib_frames2
    
    def save_calibration_videos(self, frames1, frames2):
        """Сохранение калибровочных видео"""
        if not frames1 or not frames2:
            return False
        
        if len(frames1) < 5:
            return False
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames1[0].shape[:2]
        
        print(f"Сохранение калибровочных видео размером {width}x{height}...")
        
        out1 = cv2.VideoWriter('calib_video1_temp.mp4', fourcc, self.fps, (width, height))
        out2 = cv2.VideoWriter('calib_video2_temp.mp4', fourcc, self.fps, (width, height))
        
        for frame in frames1:
            out1.write(frame)
        for frame in frames2:
            out2.write(frame)
        
        out1.release()
        out2.release()
        
        return True
    
    def initialize_stitcher(self) -> bool:
        """Инициализация сшивателя"""
        calib_frames1, calib_frames2 = self.collect_calibration_frames()
        
        if calib_frames1 is None or calib_frames2 is None:
            return False
        
        if not self.save_calibration_videos(calib_frames1, calib_frames2):
            return False
        
        print(f"Инициализация сшивателя (GPU: {'включено' if self.use_gpu else 'выключено'})...")
        
        self.stitcher = OptimizedCylindricalStitcher(
            video1_path='calib_video1_temp.mp4',
            video2_path='calib_video2_temp.mp4',
            output_path='temp_output',
            num_calibration_frames=min(5, len(calib_frames1)),
            neutral_plane_t=self.neutral_plane_t,
            fov_horizontal=self.fov_horizontal,
            use_gpu=self.use_gpu
        )
        
        if self.stitcher.gpu_available:
            print("GPU ускорение доступно и активно")
        else:
            print("GPU ускорение недоступно, работает CPU версия")
        
        self.stitcher.initialize_stitching_parameters()
        
        test_frame1 = calib_frames1[0]
        test_frame2 = calib_frames2[0]
        test_stitched = self.stitcher.stitch_frame(test_frame1, test_frame2)
        
        self.stitcher.cylindrical_map_x, self.stitcher.cylindrical_map_y = \
            self.stitcher.create_cylindrical_map(
                self.stitcher.output_size[0], self.stitcher.output_size[1]
            )
        
        self.stitcher.analyze_and_compute_crop(test_stitched)
        
        if os.path.exists('calib_video1_temp.mp4'):
            os.remove('calib_video1_temp.mp4')
        if os.path.exists('calib_video2_temp.mp4'):
            os.remove('calib_video2_temp.mp4')
        
        print(f"Сшиватель инициализирован. Финальный размер: {self.stitcher.final_output_size}")
        return True
    
    def initialize_video_writer(self) -> bool:
        """Инициализация VideoWriter"""
        self.output_width = self.stitcher.final_output_size[0]
        self.output_height = self.stitcher.final_output_size[1]
        
        if self.output_width % 2 != 0:
            self.output_width += 1
        if self.output_height % 2 != 0:
            self.output_height += 1
        
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
    
    def process_frame_pair(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Обработка пары кадров"""
        frame1_undistorted = self.calibration.undistort_image(frame1)
        frame2_undistorted = self.calibration.undistort_image(frame2)
        
        stitched = self.stitcher.stitch_frame(frame1_undistorted, frame2_undistorted)
        cylindrical = self.stitcher.cylindrical_projection(stitched)
        cropped_frame = self.stitcher.apply_crop(cylindrical)
        stitched_frame = self.stitcher.ensure_even_size(
            cropped_frame, self.stitcher.final_output_size
        )
        
        people = self.detector.process_frame(stitched_frame)
        result_frame = self.detector.drawing(stitched_frame, people)
        
        return result_frame
    
    def run(self):
        """Основной цикл обработки"""
        print("\n=== Запуск обработки видеопотоков ===")
        print(f"GPU ускорение: {'включено' if self.use_gpu else 'выключено'}")
        print("Нажмите 'q' для выхода")
        print("Нажмите 'p' для паузы/продолжения")
        print("Нажмите 'g' для переключения GPU/CPU (в режиме реального времени)")
        
        if not self.initialize_video_streams():
            print("Ошибка инициализации видеопотоков")
            return
        
        if not self.initialize_stitcher():
            print("Ошибка инициализации сшивателя")
            return
        
        if not self.initialize_video_writer():
            print("Ошибка инициализации VideoWriter")
            return
        
        window_name = "Сшитое видео с детекцией людей"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.output_width // 2, self.output_height // 2)
        
        self.start_time = time.time()
        last_stat_time = self.start_time
        stat_stitch_times = []
        
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
                elif key == ord('g') and not is_paused:
                    self.stitcher.gpu_enabled = not self.stitcher.gpu_enabled
                    status = "включено" if self.stitcher.gpu_enabled else "выключено"
                    print(f"GPU ускорение: {status}")
                
                if is_paused:
                    time.sleep(0.1)
                    continue
                
                ret1, frame1 = self.cap1.read()
                ret2, frame2 = self.cap2.read()
                
                if not ret1 or not ret2:
                    print("Достигнут конец видео или ошибка чтения")
                    break
                
                self.frame_counter += 1
                
                if self.skip_frames > 0 and self.frame_counter % (self.skip_frames + 1) != 0:
                    if hasattr(self, 'last_result_frame'):
                        cv2.imshow(window_name, self.last_result_frame)
                    continue
                
                stitch_start = time.time()
                result_frame = self.process_frame_pair(frame1, frame2)
                stitch_time = time.time() - stitch_start
                stat_stitch_times.append(stitch_time)
                
                self.last_result_frame = result_frame
                cv2.imshow(window_name, result_frame)
                self.video_writer.write(result_frame)
                self.saved_frames += 1
                
                current_time = time.time()
                if current_time - last_stat_time >= 5.0:
                    elapsed = current_time - self.start_time
                    fps = self.saved_frames / elapsed if elapsed > 0 else 0
                    
                    if stat_stitch_times:
                        avg_stitch = np.mean(stat_stitch_times[-100:]) * 1000 if stat_stitch_times else 0
                        max_stitch = np.max(stat_stitch_times[-100:]) * 1000 if stat_stitch_times else 0
                    else:
                        avg_stitch = max_stitch = 0
                    
                    print(f"Обработано: {self.saved_frames} кадров, FPS: {fps:.1f}, "
                          f"Сшивка: {avg_stitch:.1f}ms (макс: {max_stitch:.1f}ms), "
                          f"GPU: {'✓' if self.stitcher.gpu_available and self.stitcher.gpu_enabled else '✗'}")
                    
                    last_stat_time = current_time
                
        except KeyboardInterrupt:
            print("Обработка прервана пользователем")
        except Exception as e:
            print(f"Ошибка в основном цикле: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Очистка ресурсов"""
        print("\nОчистка ресурсов...")
        
        if hasattr(self, 'stitcher') and self.stitcher:
            gpu_status = "да" if self.stitcher.gpu_available else "нет"
            gpu_used = "да" if (self.stitcher.gpu_available and self.stitcher.gpu_enabled) else "нет"
            print(f"GPU доступно: {gpu_status}, Использовалось: {gpu_used}")
        
        if self.cap1 and self.cap1.isOpened():
            self.cap1.release()
            print("Видеопоток 1 освобожден")
        
        if self.cap2 and self.cap2.isOpened():
            self.cap2.release()
            print("Видеопоток 2 освобожден")
        
        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
            print(f"VideoWriter закрыт. Видео сохранено в: {self.output_path}")
            
            if os.path.exists(self.output_path):
                file_size = os.path.getsize(self.output_path) / (1024 * 1024)
                print(f"Размер выходного файла: {file_size:.2f} MB")
        
        cv2.destroyAllWindows()
        
        if self.start_time:
            total_time = time.time() - self.start_time
            avg_fps = self.saved_frames / total_time if total_time > 0 else 0
            
            print("\n=== Финальная статистика ===")
            print(f"Всего кадров в потоке: {self.frame_counter}")
            print(f"Сохранено кадров: {self.saved_frames}")
            print(f"Пропущено кадров: {self.frame_counter - self.saved_frames}")
            print(f"Общее время: {total_time:.1f} секунд")
            print(f"Средний FPS: {avg_fps:.1f}")
            print(f"Размер видео: {self.output_width}x{self.output_height}")
            if hasattr(self, 'stitcher') and self.stitcher:
                print(f"GPU использовалось: {'да' if (self.stitcher.gpu_available and self.stitcher.gpu_enabled) else 'нет'}")
        
        print("Обработка завершена")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Обработка видеопотоков')
    parser.add_argument('--config', default='config.yaml', help='Конфигурационный файл')
    parser.add_argument('--calibration', default='calibration_results.npz', 
                       help='Файл калибровки камеры')
    parser.add_argument('--output', help='Выходной файл')
    parser.add_argument('--no-gpu', action='store_true', help='Отключить GPU ускорение')
    
    args = parser.parse_args()
    
    try:
        processor = RealTimeVideoProcessor(args.config, args.calibration)
        
        if args.output:
            processor.output_path = args.output
        
        if args.no_gpu:
            processor.use_gpu = False
            
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