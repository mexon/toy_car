# This script calculates an 8-way normalised hue histogram based on a set of training images

import numpy as np
import cv2
import numpy
import os
import sys

def calculate_histogram(filename):
    img = cv2.imread(filename)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    saturation = cv2.divide(hsv[:,:,1], 16)
    lightness = cv2.divide(hsv[:,:,2], 16)

    color_signal = cv2.multiply(saturation, lightness)
    max_signal = cv2.minMaxLoc(color_signal)[1]
    ret,color_signal = cv2.threshold(color_signal, max_signal / 4, 255, cv2.THRESH_BINARY)

    blur_size = int(max(img.shape[0], img.shape[1]) / 80)
    blur_kernel = numpy.ones((blur_size,blur_size),numpy.float32)/blur_size/blur_size
    blurred = cv2.filter2D(color_signal, -1, blur_kernel)

    max_diff = cv2.minMaxLoc(blurred)[1]
    ret, thresholded = cv2.threshold(blurred, max_diff / 2, 255, cv2.THRESH_BINARY)

    histogram = cv2.calcHist([hsv], [1], thresholded, [8], [0,256])
    histogram_norm = np.linalg.norm(histogram)
    histogram_normalised = histogram / histogram_norm
    return histogram_normalised

training_image_directory = sys.argv[1]
if not os.path.exists(training_image_directory):
    print(f"Image training directory not found: {training_image_directory}")
    exit()

histograms = []
for file in os.listdir(training_image_directory):
    filename = os.fsdecode(file)
    histograms.append(calculate_histogram(f"{training_image_directory}/{filename}"))

average = [round(float(i[0]), 2) for i in np.average(histograms, axis=0)]
print(f"{average}")
