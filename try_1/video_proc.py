import cv2
import yaml
from ultralytics import YOLO
import numpy as np
import logging
import sys
import time

class Detection:
    def __init__(self, config):
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
            
            required_params = ['rtsp_url', 'model_path', 'confidence', 'skip_frames', 'scale', 'save_path']
            for param in required_params:
                if param not in data:
                    raise ValueError(f"Missing required configuration parameter: {param}")
            
            if not isinstance(data['skip_frames'], int) or data['skip_frames'] < 0:
                raise ValueError("skip_frames must be a non-negative integer")
            
            if not isinstance(data['confidence'], (int, float)) or not (0 <= data['confidence'] <= 1):
                raise ValueError("confidence must be a float between 0 and 1")
            self.logger.info("Configuration loaded successfully")
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading configuration: {e}")
            raise

        self.url = data['rtsp_url']
        self.confidence = data['confidence']
        self.skip_f = data['skip_frames']
        self.scale = data['scale']
        self.height = 0
        self.weight = 0
        self.save_path = data['save_path']

        try:
            self.model = YOLO(data['model_path'])
            self.logger.info(f"Model loaded successfully from {data['model_path']}")

        except Exception as e:
            self.logger.error(f"Error loading model from {data['model_path']}: {e}")
            raise


    def process_frame(self, frame):
        # Process frame to detect people
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame provided for processing")
                
            results = self.model(frame, conf=self.confidence, classes=[0])[0]
            people = []

            if results.boxes is not None and len(results.boxes) > 0:
                coords = results.boxes.xyxy.cpu().numpy().astype(np.int32)
                
                for coord, conf in zip(coords, results.boxes.conf):
                    x1, y1, x2, y2 = coord
                    people.append((x1, y1, x2, y2, conf))
            
            return people
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return []

    def drawing(self, frame, people):
        # Draw bounding boxes and labels on frame
        if not people:
            return frame
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame provided for drawing")
                
            for (x1, y1, x2, y2, conf) in people:
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    self.logger.warning(f"Invalid coordinates: ({x1}, {y1}, {x2}, {y2}) for frame shape {frame.shape}")
                    continue
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{conf:.2f}, ({x1}, {y1}) : ({x2}, {y2})"
                cv2.putText(frame, text, (x1, y2 + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return frame
            
        except Exception as e:
            self.logger.error(f"Error drawing on frame: {e}")
            return frame

    def resize_frame(self):
        # Resize frame
        try:                
            new_width = int(self.width * self.scale)
            new_height = int(self.height * self.scale)
            
            if new_width <= 0 or new_height <= 0:
                raise ValueError("Invalid dimensions after resizing")
                
            return (new_width, new_height)
            
        except Exception as e:
            self.logger.error(f"Error resizing frame: {e}")
            return (self.width, self.height)

    def process_rtsp(self):
        # RTSP video processing function        
        try:
            self.logger.info(f"Attempting to connect to RTSP stream: {self.url}")
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                raise ConnectionError(f"Failed to connect to RTSP stream: {self.url}")
            
            ret, test_frame = self.cap.read()
            if not ret:
                raise ValueError("Failed to read test frame from RTSP stream")
                
            self.height, self.width = test_frame.shape[:2]
            self.logger.info(f"Video stream properties: {self.width}x{self.height}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.save_path, fourcc, 20.0, (self.width, self.height))
            
            if not self.out.isOpened():
                raise IOError("Failed to initialize video writer")
                
            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
            self.logger.info("Starting video processing...")
            times = []

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from stream")
                    break
                
                self.frame_count += 1
                if self.frame_count % (self.skip_f + 1) != 0:
                    self.out.write(frame)
                    small_frame = cv2.resize(frame, self.resize_frame())
                    cv2.resizeWindow('Video', small_frame.shape[1], small_frame.shape[0])
                    cv2.imshow('Video', small_frame)
                    continue
                
                try:
                    start = time.time()
                    people = self.process_frame(frame)
                    end = time.time()
                    times.append((end - start) * 1000)
                    frame_with_detections = self.drawing(frame, people)
                    self.out.write(frame_with_detections)
                    
                    small_frame = cv2.resize(frame, self.resize_frame())
                    cv2.resizeWindow('Video', small_frame.shape[1], small_frame.shape[0])
                    cv2.imshow('Video', small_frame)
                    
                    
                
                except Exception as frame_error:
                    self.logger.error(f"Error processing frame {self.frame_count}: {frame_error}")
                    continue

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("Exit command received")
                    break

        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in RTSP processing: {e}")
        finally:
            self.logger.info("Cleaning up resources...")
            
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.logger.info("Video capture released")
                
            if self.out and self.out.isOpened():
                self.out.release()
                self.logger.info("Video writer released")
                
            cv2.destroyAllWindows()
            self.logger.info(f"Processing completed")
            self.logger.info(f"Average total time: {sum(times)/(self.frame_count):.2f} ms")