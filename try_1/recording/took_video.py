import cv2
import threading
import time
from datetime import datetime
import yaml
import os

class Record:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.cameras = config['videos']
        self.recording = False
        self.writers = {}
        os.makedirs('videos', exist_ok=True)
        
    def record_camera(self, cam):
        name = cam['name']
        url = cam['url']
        
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"Не удалось подключиться к {name}")
            return
            
        while not self.recording:
            time.sleep(0.01)
            
        filename = f"videos/{name}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        print(f"Начало записи {name}")
        
        while self.recording:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                
        out.release()
        cap.release()
        print(f"Запись {name} сохранена: {filename}")
        
    def start(self):
        print(f"\nЗагружено камер: {len(self.cameras)}")
        
        threads = []
        for cam in self.cameras:
            t = threading.Thread(target=self.record_camera, args=(cam,))
            t.daemon = True
            threads.append(t)
            t.start()
            
        print("Подключение к камерам...")
        time.sleep(3)
        
        self.recording = True
        print(f"\nЗапись начата")
        print("Нажмите Enter для остановки...\n")
        
        input()
        self.recording = False
        print(f"\nЗапись остановлена")
        
        for t in threads:
            t.join(timeout=2)
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recorder = Record('conf.yaml')
    recorder.start()
