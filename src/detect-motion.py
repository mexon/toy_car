import cv2
import numpy as np
import os
import time
import socket
import math
import sys

# ==========================================
# 1. UDP 網路設定 (作業規格: Port 5000)
# ==========================================
UDP_IP = "127.0.0.1"  # 傳送到本機，若其他電腦要接收可改為對應 IP
UDP_PORT = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ==========================================
# 2. 影片路徑處理 (防範路徑與檔名錯誤)
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = sys.argv[1]

if not os.path.exists(video_path):
    print(f"錯誤：找不到檔案 '{video_path}'，請確認影片檔名稱與位置！")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("錯誤：無法開啟影片。")
    exit()

# ==========================================
# 3. 2D 像素 -> 3D/2D 真實世界座標校正 (Homography)
# ==========================================
# 請根據你畫面中場地四個角的像素點進行調整 (u, v)
# 此處以標準 1280x720 畫面範例角點為例：
pts_image = np.array([
    [100, 100],   # 左上角像素 (u, v)
    [1180, 100],  # 右上角像素 (u, v)
    [1180, 620],  # 右下角像素 (u, v)
    [100, 620]    # 左下角像素 (u, v)
], dtype=np.float32)

# 對應的真實世界 2D 座標 (單位: mm，場地 2.5m x 1.5m = 2500mm x 1500mm)
pts_real = np.array([
    [0.0, 0.0],          # 左上角
    [2500.0, 0.0],       # 右上角
    [2500.0, 1500.0],    # 右下角
    [0.0, 1500.0]        # 左下角
], dtype=np.float32)

# 計算單應性矩陣 H
H_matrix, _ = cv2.findHomography(pts_image, pts_real)

# Car histograms computed by script calculate-histogram-for-car.py
car_hue_histograms = {
    'car1' : [0.0, 0.21, 0.51, 0.53, 0.44, 0.3, 0.15, 0.18],
    'car2' : [0.05, 0.07, 0.09, 0.15, 0.38, 0.78, 0.29, 0.31],
}

def image_to_real(u, v, H):
    """將影像像素 (u, v) 轉換為真實世界 (X, Y) mm"""
    pt = np.array([u, v, 1.0], dtype=np.float32).reshape(3, 1)
    real_pt = np.dot(H, pt)
    real_pt /= real_pt[2]
    return float(real_pt[0][0]), float(real_pt[1][0])


# ==========================================
# 4. 賽車追蹤與狀態計算類別
# ==========================================
class ToyCarTracker:
    def __init__(self, car_name="Red Racer"):
        self.car_name = car_name
        self.previous_frame = None
        self.blur_kernel = np.ones((40, 40), np.float32) / 1600.0

        # 歷史狀態 (用於計算速度與角速度)
        self.prev_car_center_real = [None, None]
        self.prev_car_orientation = None
        self.prev_time_us = None

        # 平滑濾波變數
        self.vx_smooth = 0.0
        self.vy_smooth = 0.0
        self.angular_velocity_smooth = 0.0

    # Recognise a car:
    # 1. Extract the hue values of the pixels in the detected car area
    # 2. Compute an 8-way histogram of hue values
    # 3. Compare by vector distance against hue histograms from training data of various cars
    # 4. Take the best match
    def recognise_car(self, frame, mask):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_histogram = cv2.calcHist([hsv_frame], [1], mask, [8], [0,256])
        frame_histogram_norm = np.linalg.norm(frame_histogram)
        frame_histogram_normalised = frame_histogram / frame_histogram_norm
        distances = [[car_name, np.linalg.norm(frame_histogram_normalised - car_histogram)]
                     for car_name, car_histogram in car_hue_histograms.items()]
        best_match = max(distances, key=lambda x:x[1])
        return best_match[0]

    def process_frame(self, frame, current_time_us):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.previous_frame is None:
            self.previous_frame = gray_frame
            return frame, None

        # 1. 運動影格相減 (Frame Differencing)
        difference = cv2.absdiff(gray_frame, self.previous_frame)
        self.previous_frame = gray_frame

        blurred = cv2.filter2D(difference, -1, self.blur_kernel)
        max_diff = cv2.minMaxLoc(blurred)[1]

        # 預設為未偵測到賽車 (-1000.0, -1000.0)
        car_detected = False
        car_name = "unrecognised"
        car_center_image_int = [-1, -1]
        car_center_real = [-1000.0, -1000.0]
        car_orientation = 0.0
        car_center_x_diff, car_center_y_diff = 0.0, 0.0
        angular_velocity = 0.0

        if max_diff >= 10:
            ret, thresholded = cv2.threshold(blurred, max_diff / 2, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(cnt) > 100:
                    car_name = self.recognise_car(frame, thresholded)
                    car_detected = True

                    # 取得最小外接矩形
                    rect = cv2.minAreaRect(cnt)
                    car_center_image, (width, height), angle = rect
                    car_center_image_int = [int(car_center_image[0]), int(car_center_image[1])]

                    # 計算真實座標 (mm)
                    car_center_real = image_to_real(car_center_image[0], car_center_image[1], H_matrix)

                    # 計算朝向角度 (car_orientation, degrees)
                    car_orientation = angle

                    # 計算速度 (car_center_x_diff, car_center_y_diff) 與 角速度 (angular_velocity)
                    if self.prev_time_us is not None:
                        dt = (current_time_us - self.prev_time_us) / 1000000.0  # 轉為秒
                        if dt > 0:
                            # 速度 = 位置變化 / 時間 (mm/s)
                            raw_dx = (car_center_real[0] - self.prev_car_center_real[0]) / dt
                            raw_dy = (car_center_real[1] - self.prev_car_center_real[1]) / dt

                            # 修正角度跨越 +/-180 度問題
                            d_theta = (car_orientation - self.prev_car_orientation + 180.0) % 360.0 - 180.0
                            raw_omega = d_theta / dt

                            # 一階指數平滑化
                            alpha = 0.3
                            self.vx_smooth = alpha * raw_dx + (1 - alpha) * self.vx_smooth
                            self.vy_smooth = alpha * raw_dy + (1 - alpha) * self.vy_smooth
                            self.angular_velocity_smooth = alpha * raw_omega + (1 - alpha) * self.angular_velocity_smooth

                            car_center_x_diff, car_center_y_diff = self.vx_smooth, self.vy_smooth
                            angular_velocity = self.angular_velocity_smooth

                    # 更新上一影格紀錄
                    self.prev_car_center_real = car_center_real
                    self.prev_car_orientation = car_orientation
                    self.prev_time_us = current_time_us

                    # 繪製車子框線與資訊
                    box = np.int64(cv2.boxPoints(rect))
                    cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (car_center_image_int[0], car_center_image_int[1]), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID: {car_name}", (car_center_image_int[0] + 10, car_center_image_int[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 封裝輸出資訊字串 (符合老師要求格式)
        # 格式: timestamp:"car_id",x,y,car_orientation,car_center_x_diff,car_center_y_diff,angular_velocity,u,w\n
        udp_output_string = (
            f'{current_time_us}:"{car_name}",'
            f'{car_center_real[0]:.1f},{car_center_real[1]:.1f},{car_orientation:.1f},'
            f'{car_center_x_diff:.1f},{car_center_y_diff:.1f},{angular_velocity:.1f},'
            f'{car_center_image_int[0]},{car_center_image_int[1]}\n'
        )

        return frame, udp_output_string


# ==========================================
# 5. 主程式迴圈
# ==========================================
tracker = ToyCarTracker(car_name="Red Racer")
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("影片播放完畢。")
        break

    # 取得微秒時間戳記 (microseconds relative clock)
    current_time_us = int((time.time() - start_time) * 1000000)

    # 處理影格
    processed_frame, output_msg = tracker.process_frame(frame, current_time_us)

    if output_msg:
        # 1. 終端機即時印出訊息
        print(output_msg, end='')

        # 2. UTF-8 編碼並透過 UDP 發送至 Port 5000
        sock.sendto(output_msg.encode('utf-8'), (UDP_IP, UDP_PORT))

    # 顯示即時畫面
    resized = cv2.resize(processed_frame, (1024, 1024))
    cv2.imshow('Global Vision Server', resized)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
