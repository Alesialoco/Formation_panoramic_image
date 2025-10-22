import cv2
import numpy as np
import configparser
from people_detected import PeopleDetector

class VideoStitcher:
    def __init__(self, config_file='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        self.detector = PeopleDetector(config_file)

        self.stitcher = None

        self.overlap_percentage = self.config.getfloat('STITCHING', 'overlap_percentage', fallback=25.0)
        self.resolution_width = self.config.getint('STITCHING', 'resolution_width', fallback=640)
        self.resolution_height = self.config.getint('STITCHING', 'resolution_height', fallback=480)
        self.stitch_mode = self.config.get('STITCHING', 'stitch_mode', fallback='SCANS')
        self.display_width = self.config.getint('OUTPUT', 'display_width', fallback=1200)
        self.display_height = self.config.getint('OUTPUT', 'display_height', fallback=600)
        self.show_individual_cameras = self.config.getboolean('OUTPUT', 'show_individual_cameras', fallback=True)

        self.confidence_threshold = self.config.getfloat('DETECTION', 'confidence', fallback=0.5)
        self.model_path = self.config.get('MODEL', 'path', fallback='yolov8n.pt')

        self.caps = []
        self.frames = []

    def initialize_stitcher(self):
        try:
            if self.stitch_mode.upper() == 'SCANS':
                self.stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
                print('Stitcher инициализирован (режим SCANS)')
            else:
                self.stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
                print('Stitcher инициализирован (режим PANORAMA)')
            return True
        except Exception as e:
            print(f'Ошибка инициализации Stitcher: {e}')
            return False

    def connect_to_streams(self):
        self.caps = []
        sources = []

        try:
            if self.config.has_section('RTSP'):
                for key, value in self.config.items('RTSP'):
                    if value.strip():
                        sources.append(value.strip())

            if len(sources) < 2:
                raise Exception('Недостаточно источников в конфиге')
        except Exception as e:
            print(f'Ошибка при подключении к источникам: {e}')
            return False

        success_count = 0
        for i, source in enumerate(sources):
            try:
                cap = cv2.VideoCapture(source)

                if cap.isOpened():
                    self.caps.append(cap)
                    success_count += 1
                    print(f'Успешно подключено к источнику {i+1}: {source}')
                else:
                    print(f'Ошибка при подключении к источнику {i+1}: {source}')
            except Exception as e:
                print(f'Исключение при подключении к источнику {i+1}: {e}')

        print(f'Успешно подключено к {len(self.caps)} из {len(sources)} источников')
        return success_count >= 2

    def read_frames(self):
        self.frames = []

        for i, cap in enumerate(self.caps):
            ret, frame = cap.read()

            if ret:
                frame = cv2.resize(frame, (self.resolution_width, self.resolution_height))
                self.frames.append(frame)
            else:
                print(f'Ошибка чтения кадра с источника {i+1}')

                black_frame = np.zeros((self.resolution_height, self.resolution_width, 3), dtype=np.uint8)
                cv2.putText(black_frame, f'Source {i+1} error', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                self.frames.append(black_frame)

        return len(self.frames) >= 2

    def stitch_frames(self):
        if self.stitcher is None:
            if not self.initialize_stitcher():
                return None

        try:
            status, panorama = self.stitcher.stitch(self.frames)

            if status == cv2.Stitcher_OK:
                print('Сшивка выполнена успешно')
                return panorama
            else:
                raise Exception(f'код {status} != {cv2.Stitcher_OK}')
        except Exception as e:
            print(f'Исключение при сшивке: {e}')
            return None

    def detect_people_in_panorama(self, panorama):
        try:
            detections = self.detector.process_frame(panorama)
            panorama_with_detections = self.detector.draw_detections(panorama.copy(), detections)

            info_text = f'People: {len(detections)} | Confidence: {self.detector.conf_threshold} | Overlap: {self.overlap_percentage}%'
            cv2.putText(panorama_with_detections, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return panorama_with_detections, detections
        except Exception as e:
            print(f'Ошибка поиска людей: {e}')
            return panorama, []

    def run(self):
        if not self.connect_to_streams():
            return

        print('Запуск панорамной сшивки. Нажмите "q" для выхода')
        print(f'Параметры: {self.resolution_width}x{self.resolution_height}, overlap: {self.overlap_percentage}%')

        try:
            while True:
                if not self.read_frames():
                    print('Ошибка чтения кадров')
                    return

                panorama = self.stitch_frames()

                if panorama is not None:
                    panorama_with_detections, detections = self.detect_people_in_panorama(panorama)
                    display_frame = cv2.resize(panorama_with_detections, (self.display_width, self.display_height))
                    cv2.imshow('Panorama - People Detection', display_frame)

                    if self.show_individual_cameras:
                        for i, frame in enumerate(self.frames):
                            cv2.imshow(f'Camera {i+1}', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print('Программа прервана пользователем')
        except Exception as e:
            print(f'Критическая ошибка:')
        finally:
            for cap in self.caps:
                cap.release()
            cv2.destroyAllWindows()
            print('Ресурсы освобождены')
