import cv2
import numpy as np

class ImageEnhancer:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def enhance_clahe(self, image):
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = self.clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            return self.clahe.apply(image)
