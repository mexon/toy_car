# This script looks for highly-saturated regions of colour in an
# image.  In practice this reliably picks up traffic cones and
# people's feet, but tends to detect an actual toy car as two separate
# blobs.

import numpy as np
import cv2
import numpy
import os

def process_image(filename):
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

    contours,hierarchy = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = img.copy()
    cv2.drawContours(annotated, contours, -1, (0,0,255), 2)

    # cv2.imshow('Frame', img)
    # key = cv2.waitKey(0)
    # cv2.imshow('Frame', blurred)
    # key = cv2.waitKey(0)

    resized = cv2.resize(annotated, (1024, 1024))
    cv2.imshow('Frame', resized)
    key = cv2.waitKey(0)

directory = "toy_car_photos"

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    process_image(f"{directory}/{filename}")
