import cv2
from ultralytics import YOLO
import configparser
import numpy as np

class PeopleDetector:
    def __init__(self, config_file='config.ini'):
        try:
            self.config = configparser.ConfigParser()
            self.config.read(config_file)

            model_path = self.config.get('MODEL', 'path', fallback='yolo8n.pt')
            print(f'Загрузка модели: {model_path}')
            self.model = YOLO(model_path)

            self.conf_threshold = self.config.getfloat('DETECTION', 'confidence', fallback=0.5)
            self.classes = [0]

            self.rtsp_url = self.config.get('RTSP', 'url')
            self.cap = None

            print('Инициализация завершена успешно')
        except Exception as e:
            print(f'Ошибка инициализации: {e}')
            raise SystemExit('Программа завершена из-за ошибки инициализации')


    def connect_to_stream(self):
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                raise Exception(f'Не удалось подключиться к RTSP потоку: {self.rtsp_url}')

            print(f'Успешно подключено к RTSP: {self.rtsp_url}')
            return True
        except Exception as e:
            print(f'Ошибка подключения: {e}')
            return False

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
            color = (0, 255, 0) # green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'Conf: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            coords_text = f'({x1},{y1})-({x2},{y2})'
            cv2.putText(frame, coords_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame

    def run(self):
        if not self.connect_to_stream():
            print('Не удалось подключиться к RTSP потоку. Завершение работы')
            return

        print('Запуск обнаружения людей. Нажмите "q" для выхода')

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print('Ошибка чтения кадра. Попытка переподключения...')
                    cv2.waitKey(1000)
                    if not self.connect_to_stream():
                        print('Переподключение не удалось. Завершение работы')
                        break
                    continue

                detections = self.process_frame(frame)
                frame_with_detections = self.draw_detections(frame.copy(), detections)
                info_text = f'People: {len(detections)} | Confidence: {self.conf_threshold}'
                cv2.putText(frame_with_detections, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('People Detection - YOLOv8', frame_with_detections)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print('Программа прервана пользователем')
        except Exception as e:
            print(f'Критическая ошибка во время выполнения: {e}')
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print('Ресурсы освобождены')
