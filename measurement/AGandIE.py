import cv2
import numpy as np

def calculate_entropy(image):
    """Calculate the entropy of an image."""
    histogram = cv2.calcHist([image], [0], None, [256], [0,256])
    histogram_length = np.sum(histogram)
    probabilities = histogram / histogram_length
    return -np.sum([p * np.log2(p) for p in probabilities if p != 0])

def calculate_average_gradient(image):
    """Calculate the average gradient of an image."""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    gradient = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    return np.mean(gradient)