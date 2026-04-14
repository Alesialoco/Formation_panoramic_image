import cv2
import yaml
from ultralytics import YOLO
import numpy as np
import logging
import sys

class Detection:
    """
    Класс для детекции объектов на видео с использованием YOLO
    """
    
    def __init__(self, config):
        """
        Инициализация класса детекции
        
        Args:
            config (str): Путь к конфигурационному файлу YAML
            
        Raises:
            ValueError: При отсутствии обязательных параметров конфигурации
            FileNotFoundError: Если конфигурационный файл не найден
            Exception: При ошибках загрузки модели
        """
        self.cap = None
        self.out = None
        self.frame_count = 0
        self.frames_processed = 0

        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)

        try:
            with open(config, 'r', encoding="utf-8") as file:
                data = yaml.safe_load(file)
            
            required_params = ['model_path', 'confidence', 'skip_frames', 'scale', 'save_path']
            for param in required_params:
                if param not in data:
                    raise ValueError(f"Отсутствует обязательный параметр конфигурации: {param}")
            
            if not isinstance(data['skip_frames'], int) or data['skip_frames'] < 0:
                raise ValueError("skip_frames должно быть целым неотрицательным числом")
            
            if not isinstance(data['confidence'], (int, float)) or not (0 <= data['confidence'] <= 1):
                raise ValueError("confidence должно быть числом с плавающей точкой от 0 до 1")
            self.logger.info("Конфигурация успешно загружена")
            
        except FileNotFoundError:
            self.logger.error(f"Конфигурационный файл не найден: {config}")
            raise
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка при загрузке конфигурации: {e}")
            raise

        self.confidence = data['confidence']
        self.skip_f = data['skip_frames']
        self.scale = data['scale']
        self.height = 0
        self.weight = 0
        self.save_path = data['save_path']

        try:
            self.model = YOLO(data['model_path'])
            self.logger.info(f"Модель успешно загружена из {data['model_path']}")

        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели из {data['model_path']}: {e}")
            raise

    def process_frame(self, frame):
        """
        Обработка кадра для детекции людей
        
        Args:
            frame (numpy.ndarray): Входной кадр
            
        Returns:
            list: Список обнаруженных людей с координатами и уверенностью
        """
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Предоставлен невалидный кадр для обработки")
                
            results = self.model(frame, conf=self.confidence, classes=[0])[0]
            people = []

            if results.boxes is not None and len(results.boxes) > 0:
                coords = results.boxes.xyxy.cpu().numpy().astype(np.int32)
                
                for coord, conf in zip(coords, results.boxes.conf):
                    x1, y1, x2, y2 = coord
                    people.append((x1, y1, x2, y2, conf))
            
            return people
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке кадра: {e}")
            return []

    def drawing(self, frame, people):
        """
        Рисование bounding boxes и меток на кадре
        
        Args:
            frame (numpy.ndarray): Исходный кадр
            people (list): Список обнаруженных людей
            
        Returns:
            numpy.ndarray: Кадр с нарисованными bounding boxes
        """
        if not people:
            return frame
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Предоставлен невалидный кадр для рисования")
                
            for (x1, y1, x2, y2, conf) in people:
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    self.logger.warning(f"Невалидные координаты: ({x1}, {y1}, {x2}, {y2}) для размера кадра {frame.shape}")
                    continue
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                text = f"{conf:.2f}, ({x1}, {y1}) : ({x2}, {y2})"
                cv2.putText(frame, text, (x1, y2 + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return frame
            
        except Exception as e:
            self.logger.error(f"Ошибка при рисовании на кадре: {e}")
            return frame

    def resize_frame(self):
        """
        Расчет новых размеров кадра для отображения
        
        Returns:
            tuple: Новые размеры (ширина, высота)
            
        Raises:
            ValueError: При некорректных размерах после масштабирования
        """
        try:                
            new_width = int(self.width * self.scale)
            new_height = int(self.height * self.scale)
            
            if new_width <= 0 or new_height <= 0:
                raise ValueError("Невалидные размеры после изменения масштаба")
                
            return (new_width, new_height)
            
        except Exception as e:
            self.logger.error(f"Ошибка при изменении размера кадра: {e}")
            return (self.width, self.height)

    def process_rtsp(self):
        """
        Основная функция обработки RTSP потока или видеофайла
        
        Обрабатывает видео, выполняет детекцию людей, записывает результат
        и отображает прогресс в реальном времени.
        
        Raises:
            ConnectionError: При невозможности подключиться к RTSP потоку
            IOError: При ошибке инициализации VideoWriter
        """
        try:
            self.logger.info(f"Попытка подключения к RTSP потоку: {self.url}")
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                raise ConnectionError(f"Не удалось подключиться к RTSP потоку: {self.url}")
            
            ret, test_frame = self.cap.read()
            if not ret:
                raise ValueError("Не удалось прочитать тестовый кадр из RTSP потока")
                
            self.height, self.width = test_frame.shape[:2]
            self.logger.info(f"Свойства видео потока: {self.width}x{self.height}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.save_path, fourcc, 20.0, (self.width, self.height))
            
            if not self.out.isOpened():
                raise IOError("Не удалось инициализировать видео запись")
                
            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
            self.logger.info("Начало обработки видео...")
            times = []

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Не удалось прочитать кадр из потока")
                    break
                
                self.frame_count += 1
                if self.frame_count % (self.skip_f + 1) != 0:
                    self.out.write(frame)
                    small_frame = cv2.resize(frame, self.resize_frame())
                    cv2.resizeWindow('Video', small_frame.shape[1], small_frame.shape[0])
                    cv2.imshow('Video', small_frame)
                    continue
                
                try:
                    people = self.process_frame(frame)
                    
                    frame_with_detections = self.drawing(frame, people)
                    self.out.write(frame_with_detections)
                    
                    small_frame = cv2.resize(frame, self.resize_frame())
                    cv2.resizeWindow('Video', small_frame.shape[1], small_frame.shape[0])
                    cv2.imshow('Video', small_frame)
                    
                except Exception as frame_error:
                    self.logger.error(f"Ошибка при обработке кадра {self.frame_count}: {frame_error}")
                    continue

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("Получена команда выхода")
                    break

        except KeyboardInterrupt:
            self.logger.info("Обработка прервана пользователем")
        except Exception as e:
            self.logger.error(f"Ошибка в обработке RTSP: {e}")
        finally:
            self.logger.info("Очистка ресурсов...")
            
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.logger.info("Видео захват освобожден")
                
            if self.out and self.out.isOpened():
                self.out.release()
                self.logger.info("Видео запись освобождена")
                
            cv2.destroyAllWindows()
            self.logger.info(f"Обработка завершена")