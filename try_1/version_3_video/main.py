import argparse
import cv2
import numpy as np
import yaml
import os
import time
from video_proc import Detection
from stich import OptimizedCylindricalStitcher, StitchingParameters


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
    """Основной класс для обработки 3 видеопотоков"""
    
    def __init__(self, config_path: str, calibration_path: str = 'calibration_results.npz',
                 mode: str = 'process', calibration_file: str = 'stitching_params.pkl'):
        with open(config_path, 'r', encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        
        required_params = ['rtsp_url_1', 'rtsp_url_2', 'rtsp_url_3', 
                          'model_path', 'confidence', 'skip_frames', 'scale', 'save_path']
        
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Отсутствует обязательный параметр конфигурации: {param}")
        
        self.video_left_url = self.config['rtsp_url_1']
        self.video_center_url = self.config['rtsp_url_2']
        self.video_right_url = self.config['rtsp_url_3']
        self.skip_frames = self.config.get('skip_frames', 1)
        self.output_path = self.config.get('save_path', 'output.mp4')
        
        self.num_calibration_frames = self.config.get('num_calibration_frames', 10)
        self.fov_horizontal = self.config.get('fov_horizontal', 150)
        
        self.adaptive_smoothness = self.config.get('adaptive_smoothness', 50.0)
        self.crop_percent = self.config.get('crop_percent', 0.30)
        
        self.mode = mode
        self.calibration_file = calibration_file
        
        print(f"Режим работы: {'калибровка' if mode == 'calibrate' else 'обработка'}")
        print(f"Параметры сшивки: кадров для калибровки={self.num_calibration_frames}, "
              f"FOV={self.fov_horizontal}°, гладкость={self.adaptive_smoothness}, "
              f"обрезка боков={self.crop_percent*100:.1f}%")
        
        self.calibration = CameraCalibration(calibration_path)
        self.detector = Detection(config_path)
        self.stitcher = None
        
        self.cap_left = None
        self.cap_center = None
        self.cap_right = None
        self.video_writer = None
        
        self.frame_counter = 0
        self.saved_frames = 0
        self.start_time = None
    
    def initialize_video_streams(self) -> bool:
        """Инициализация видеопотоков"""
        self.cap_left = cv2.VideoCapture(self.video_left_url)
        self.cap_center = cv2.VideoCapture(self.video_center_url)
        self.cap_right = cv2.VideoCapture(self.video_right_url)
        
        if not self.cap_left.isOpened():
            print(f"Ошибка: Не удалось открыть левый видеопоток")
            return False
        
        if not self.cap_center.isOpened():
            print(f"Ошибка: Не удалось открыть центральный видеопоток")
            self.cap_left.release()
            return False
        
        if not self.cap_right.isOpened():
            print(f"Ошибка: Не удалось открыть правый видеопоток")
            self.cap_left.release()
            self.cap_center.release()
            return False
        
        fps_left = self.cap_left.get(cv2.CAP_PROP_FPS)
        fps_center = self.cap_center.get(cv2.CAP_PROP_FPS)
        fps_right = self.cap_right.get(cv2.CAP_PROP_FPS)
        self.fps = min(fps_left, fps_center, fps_right)
        
        if self.fps <= 0:
            self.fps = 30.0
            print(f"Не удалось определить FPS, использую {self.fps}")
        else:
            print(f"Частота кадров: {self.fps} FPS")
        
        ret_left, test_frame_left = self.cap_left.read()
        ret_center, test_frame_center = self.cap_center.read()
        ret_right, test_frame_right = self.cap_right.read()
        
        if not ret_left or not ret_center or not ret_right:
            print("Ошибка: Не удалось прочитать тестовые кадры")
            return False
        
        print(f"Размер левого потока: {test_frame_left.shape[1]}x{test_frame_left.shape[0]}")
        print(f"Размер центрального потока: {test_frame_center.shape[1]}x{test_frame_center.shape[0]}")
        print(f"Размер правого потока: {test_frame_right.shape[1]}x{test_frame_right.shape[0]}")
        
        self.cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap_center.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return True
    
    def collect_calibration_frames(self):
        """Сбор кадров для калибровки сшивателя"""
        print(f"Сбор {self.num_calibration_frames} кадров для калибровки...")
        
        calib_frames_left = []
        calib_frames_center = []
        calib_frames_right = []
        
        for i in range(self.num_calibration_frames):
            ret_left, frame_left = self.cap_left.read()
            ret_center, frame_center = self.cap_center.read()
            ret_right, frame_right = self.cap_right.read()
            
            if not ret_left or not ret_center or not ret_right:
                print(f"Предупреждение: собрано только {i} кадров из {self.num_calibration_frames}")
                break
            
            frame_left_undistorted = self.calibration.undistort_image(frame_left)
            frame_center_undistorted = self.calibration.undistort_image(frame_center)
            frame_right_undistorted = self.calibration.undistort_image(frame_right)
            
            calib_frames_left.append(frame_left_undistorted)
            calib_frames_center.append(frame_center_undistorted)
            calib_frames_right.append(frame_right_undistorted)
            
            if (i + 1) % 5 == 0:
                print(f"  Собрано кадров: {i + 1}/{self.num_calibration_frames}")
        
        self.cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap_center.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        if len(calib_frames_left) < 5:
            print(f"Ошибка: недостаточно кадров для калибровки ({len(calib_frames_left)} из 5)")
            return None, None, None
        
        print(f"Собрано {len(calib_frames_left)} кадров для калибровки")
        return calib_frames_left, calib_frames_center, calib_frames_right
    
    def save_calibration_videos(self, frames_left, frames_center, frames_right):
        """Сохранение калибровочных видео"""
        if not frames_left or not frames_center or not frames_right:
            return False
        
        if len(frames_left) < 5:
            return False
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames_left[0].shape[:2]
        
        print(f"Сохранение калибровочных видео размером {width}x{height}...")
        
        out_left = cv2.VideoWriter('calib_video_left_temp.mp4', fourcc, self.fps, (width, height))
        out_center = cv2.VideoWriter('calib_video_center_temp.mp4', fourcc, self.fps, (width, height))
        out_right = cv2.VideoWriter('calib_video_right_temp.mp4', fourcc, self.fps, (width, height))
        
        for frame in frames_left:
            out_left.write(frame)
        for frame in frames_center:
            out_center.write(frame)
        for frame in frames_right:
            out_right.write(frame)
        
        out_left.release()
        out_center.release()
        out_right.release()
        
        return True
    
    def run_calibration(self):
        """Запуск режима калибровки"""
        print("\n=== Запуск калибровки сшивки ===")
        
        if not self.initialize_video_streams():
            print("Ошибка инициализации видеопотоков")
            return
        
        calib_frames_left, calib_frames_center, calib_frames_right = self.collect_calibration_frames()
        
        if calib_frames_left is None or calib_frames_center is None or calib_frames_right is None:
            print("Ошибка сбора кадров для калибровки")
            return
        
        if not self.save_calibration_videos(calib_frames_left, calib_frames_center, calib_frames_right):
            print("Ошибка сохранения калибровочных видео")
            return
        
        print("Инициализация сшивателя для калибровки...")
        self.stitcher = OptimizedCylindricalStitcher(
            video_left_path='calib_video_left_temp.mp4',
            video_center_path='calib_video_center_temp.mp4',
            video_right_path='calib_video_right_temp.mp4',
            output_path='temp_output',
            num_calibration_frames=min(5, len(calib_frames_left)),
            fov_horizontal=self.fov_horizontal,
            adaptive_smoothness=self.adaptive_smoothness,
            crop_percent=self.crop_percent
        )
        
        self.stitcher.calibrate(self.calibration_file)
        
        for f in ['calib_video_left_temp.mp4', 'calib_video_center_temp.mp4', 'calib_video_right_temp.mp4']:
            if os.path.exists(f):
                os.remove(f)
        
        if self.cap_left:
            self.cap_left.release()
        if self.cap_center:
            self.cap_center.release()
        if self.cap_right:
            self.cap_right.release()
        
        print(f"\nКалибровка завершена успешно!")
        print(f"Параметры сохранены в: {self.calibration_file}")
    
    def initialize_stitcher_from_file(self) -> bool:
        """Инициализация сшивателя из файла параметров"""
        if not os.path.exists(self.calibration_file):
            print(f"Ошибка: Файл параметров не найден: {self.calibration_file}")
            print("Сначала выполните калибровку с параметром --mode calibrate")
            return False
        
        print(f"Загрузка параметров из {self.calibration_file}...")
        params = StitchingParameters.load(self.calibration_file)
        
        self.stitcher = OptimizedCylindricalStitcher(
            fov_horizontal=params.fov_horizontal,
            adaptive_smoothness=params.adaptive_smoothness,
            crop_percent=params.crop_percent
        )
        
        self.stitcher.set_parameters(params)
        
        print(f"Сшиватель инициализирован. Финальный размер: {self.stitcher.final_output_size}")
        return True
    
    def initialize_video_writer(self) -> bool:
        """Инициализация VideoWriter"""
        self.output_width = self.stitcher.final_output_size[0]
        self.output_height = self.stitcher.final_output_size[1]
        
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
    
    def process_frame_pair(self, frame_left: np.ndarray, frame_center: np.ndarray, 
                          frame_right: np.ndarray) -> np.ndarray:
        """Обработка трех кадров"""
        frame_left_undistorted = self.calibration.undistort_image(frame_left)
        frame_center_undistorted = self.calibration.undistort_image(frame_center)
        frame_right_undistorted = self.calibration.undistort_image(frame_right)
        
        processed_frame = self.stitcher.process_with_params(
            frame_left_undistorted, frame_center_undistorted, frame_right_undistorted)
        
        people = self.detector.process_frame(processed_frame)
        result_frame = self.detector.drawing(processed_frame, people)
        
        return result_frame
    
    def run_processing(self):
        """Основной цикл обработки"""
        print("\n=== Запуск обработки видеопотоков ===")
        print("Нажмите 'q' для выхода")
        print("Нажмите 'p' для паузы/продолжения")
        
        if not self.initialize_video_streams():
            print("Ошибка инициализации видеопотоков")
            return
        
        if not self.initialize_stitcher_from_file():
            return
        
        if not self.initialize_video_writer():
            print("Ошибка инициализации VideoWriter")
            return
        
        window_name = "Сшитое видео с детекцией людей"
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
                
                if is_paused:
                    time.sleep(0.1)
                    continue
                
                ret_left, frame_left = self.cap_left.read()
                ret_center, frame_center = self.cap_center.read()
                ret_right, frame_right = self.cap_right.read()
                
                if not ret_left or not ret_center or not ret_right:
                    print("Достигнут конец видео или ошибка чтения")
                    break
                
                self.frame_counter += 1
                
                if self.skip_frames > 0 and self.frame_counter % (self.skip_frames + 1) != 0:
                    if hasattr(self, 'last_result_frame'):
                        cv2.imshow(window_name, self.last_result_frame)
                    continue
                
                result_frame = self.process_frame_pair(frame_left, frame_center, frame_right)
                self.last_result_frame = result_frame
                cv2.imshow(window_name, result_frame)
                self.video_writer.write(result_frame)
                self.saved_frames += 1
                
                current_time = time.time()
                if current_time - last_stat_time >= 5.0:
                    elapsed = current_time - self.start_time
                    fps = self.saved_frames / elapsed if elapsed > 0 else 0
                    print(f"Обработано: {self.saved_frames} кадров, FPS: {fps:.1f}")
                    last_stat_time = current_time
                
        except KeyboardInterrupt:
            print("Обработка прервана пользователем")
        except Exception as e:
            print(f"Ошибка в основном цикле: {e}")
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
        
        if self.cap_left and self.cap_left.isOpened():
            self.cap_left.release()
            print("Левый видеопоток освобожден")
        
        if self.cap_center and self.cap_center.isOpened():
            self.cap_center.release()
            print("Центральный видеопоток освобожден")
        
        if self.cap_right and self.cap_right.isOpened():
            self.cap_right.release()
            print("Правый видеопоток освобожден")
        
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
            print(f"Всего кадров в потоке: {self.frame_counter}")
            print(f"Сохранено кадров: {self.saved_frames}")
            print(f"Пропущено кадров: {self.frame_counter - self.saved_frames}")
            print(f"Общее время: {total_time:.1f} секунд")
            print(f"Средний FPS: {avg_fps:.1f}")
            print(f"Размер видео: {self.output_width}x{self.output_height}")
        
        print("Обработка завершена")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Обработка видеопотоков (3 камеры)')
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
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())