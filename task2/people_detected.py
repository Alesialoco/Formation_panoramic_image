import cv2
from ultralytics import YOLO
import configparser
import numpy as np

class PeopleDetector:
    def __init__(self, config_file='config.ini'):
        try:
            self.config = configparser.ConfigParser()
            self.config.read(config_file)

            model_path = self.config.get('MODEL', 'path', fallback='yolov8n.pt')
            print(f'Загрузка модели: {model_path}')
            self.model = YOLO(model_path)

            self.conf_threshold = self.config.getfloat('DETECTION', 'confidence', fallback=0.5)
            self.classes = [0]

            self.cap = None

            print('PeopleDetector инициализирован успешно')
        except Exception as e:
            print(f'Ошибка инициализации PeopleDetector: {e}')
            raise

    def process_frame(self, frame):
        try:
            results = self.model(frame, conf=self.conf_threshold, classes=self.classes, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        detections.append((x1, y1, x2, y2, confidence))
            return detections

        except Exception as e:
            print(f'Ошибка обработки кадра: {e}')
            return []

    def draw_detections(self, frame, detections):
        for x1, y1, x2, y2, confidence in detections:
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'Conf: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            coords_text = f'({x1},{y1})-({x2},{y2})'
            cv2.putText(frame, coords_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame
