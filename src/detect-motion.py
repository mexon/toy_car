# This script detects the most prominent motion blob in a video.  It
# attempts to fit a rectangle around the blob, which produces a rough
# orientation.  It paints the video frame-by-frame with the rectangle
# around the detected shape.

#import numpy as np
import cv2
import numpy

cap = cv2.VideoCapture('toycar-clip.mp4')
#img = cv2.imread("test.jpg")

class ToyCarTracker:
    def __init__(self):
        self.previous_frame = None
        self.blur_kernel = numpy.ones((40,40),numpy.float32)/40/40
        
    def process_frame(self, frame):
        if self.previous_frame is None:
            self.previous_frame = frame
            return frame
        
        color_difference = cv2.absdiff(frame, self.previous_frame)
        difference = cv2.cvtColor(color_difference, cv2.COLOR_BGR2GRAY)
        self.previous_frame = frame

        blurred = cv2.filter2D(difference,-1,self.blur_kernel)
        max_difference = cv2.minMaxLoc(blurred)[1]
        ret,thresholded = cv2.threshold(blurred, max_difference / 2, 255, cv2.THRESH_BINARY)

        contours,hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = frame.copy()

        # Better to choose contour with greatest area
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = numpy.int64(box)
        cv2.drawContours(result,[box],-1,(0,0,255),2)
        
        return result

tracker = ToyCarTracker()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame = tracker.process_frame(frame)
    cv2.imshow('Frame', processed_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
