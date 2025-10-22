import cv2
import numpy as np
import pickle
import os

class CameraCalibrator:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = False

    def load_calibration(self, calibration_file):
        if os.path.exists(calibration_file):
            try:
                with open(calibration_file, 'rb') as f:
                    data = pickle.load(f)
                    self.camera_matrix = data['camera_matrix']
                    self.dist_coeffs = data['dist_coeffs']
                    self.calibrated = True
                    print(f"Калибровка загружена из {calibration_file}")
                    return True
            except Exception as e:
                print(f"Ошибка загрузки калибровки: {e}")
        return False

    def undistort_image(self, image):
        if not self.calibrated:
            return image

        h, w = image.shape[:2]

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))

        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)

        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        return undistorted
