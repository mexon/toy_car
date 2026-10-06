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
    'yellow/blue' : [0.01, 0.16, 0.55, 0.56, 0.29, 0.12, 0.03, 0.02],
    'red/blue' : [0.01, 0.02, 0.09, 0.22, 0.39, 0.48, 0.41, 0.50],
}

def image_to_real(u, v, H):
    """將影像像素 (u, v) 轉換為真實世界 (X, Y) mm"""
    pt = np.array([u, v, 1.0], dtype=np.float32).reshape(3, 1)
    real_pt = np.dot(H, pt)
    real_pt /= real_pt[2]
    return float(real_pt[0][0]), float(real_pt[1][0])

def image_coords_to_real(coords, H):
    return image_to_real(coords[0], coords[1], H)

def resize_frame(frame, max_width, max_height):
    (frame_width, frame_height, color_depth) = frame.shape
    target_size = np.array((frame_height, frame_width))
    if max_width < target_size[0]:
        target_size = np.multiply(target_size, max_width / target_size[0])
    if max_height < target_size[1]:
        target_size = np.multiply(target_size, max_height / target_size[1])
    target_size = target_size.astype(int)
    return cv2.resize(frame, target_size)


class CarData:
    alpha = 0.3

    def __init__(self):
        self.car_name = "unrecognised"

        # 歷史狀態 (用於計算速度與角速度)
        self.prev_car_center_real = None
        self.prev_car_heading = None
        self.prev_time_us = None

        # 平滑濾波變數
        self.velocity_smooth = (0.0, 0.0)
        self.angular_velocity_smooth = 0.0

    def smooth_scalar(self, new_value, old_value):
        return self.alpha * new_value + (1.0 - self.alpha) * old_value

    def smooth_vector(self, new_value, old_value):
        return [self.smooth_scalar(new, old) for new, old in zip(new_value, old_value)]

    def annotate_image(self, frame, rect, car_center_image_int, car_name, heading, blue_center):
        box = np.int64(cv2.boxPoints(rect))
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
        centre_x, centre_y = car_center_image_int
        cv2.circle(frame, (centre_x, centre_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"ID: {car_name}", (centre_x + 10, centre_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 1)
        cv2.putText(frame, f"Heading: {heading:.1f}", (centre_x + 10, centre_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

        # 藍色：偵測到的車頭 marker
        if blue_center is not None:
            cv2.circle(frame, blue_center, 8, (255, 0, 0), 2)

        # 紫色箭頭：最終 Heading
        arrow_length = 80
        theta = math.radians(heading)
        end_x = int(centre_x + arrow_length * math.cos(theta))
        end_y = int(centre_y + arrow_length * math.sin(theta))
        cv2.arrowedLine(
            frame, (centre_x, centre_y), (end_x, end_y), (255, 0, 255), 3, tipLength=0.3
        )

    # Angle changes must be with +/- 180°
    def normalise_angle_change(self, angle):
        return (angle + 180.0) % 360.0 - 180.0

    # ======================================
    # minAreaRect 車身長軸
    # ======================================
    def get_body_angle(self, rect):
        center, (width, height), angle = rect

        if width < height:

            angle += 90.0

        return angle % 180.0

    # =====================================
    # 移動方向 fallback
    # 如果藍色偵測不到
    # 才使用這個方法
    # ======================================

    def determine_heading_from_motion(self, body_angle, car_center_real):
        if self.prev_car_center_real is None:
            if self.prev_car_heading is not None:
                return self.prev_car_heading
            return body_angle

        dx = car_center_real[0] - self.prev_car_center_real[0]
        dy = car_center_real[1] - self.prev_car_center_real[1]

        distance = math.hypot(dx, dy)
        # 幾乎沒動

        if distance < 5.0:
            if self.prev_car_heading is not None:
                return self.prev_car_heading
            return body_angle

        motion_angle = math.degrees(math.atan2(dy, dx)) % 360.0
        heading1 = body_angle
        heading2 = (body_angle + 180.0) % 360.0
        diff1 = abs(self.normalise_angle_change(heading1 - motion_angle))
        diff2 = abs(self.normalise_angle_change(heading2 - motion_angle))
        if diff1 <= diff2:
            return heading1

        return heading2

    # ======================================
    # ★ 藍色車頭偵測
    # ======================================
    def detect_blue_front(self, frame, rect):
        # ----------------------------------
        # BGR -> HSV
        # ----------------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # ==================================
        # 藍色 HSV 範圍
        # OpenCV Hue：
        # 0 ~ 179
        # 這組值之後可以依影片調整
        # ==================================
        lower_blue = np.array([105, 35, 35], dtype=np.uint8)
        upper_blue = np.array([135, 130, 140], dtype=np.uint8)
        # ----------------------------------
        # 找所有藍色
        # ----------------------------------
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # ==================================
        # 建立車子區域 Mask
        # 只允許搜尋綠色框附近的藍色
        # ==================================
        car_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        cv2.fillConvexPoly(car_mask, box, 255)
        # ----------------------------------
        # 稍微放大搜尋區域
        # 因為 frame difference 的框
        # 不一定完整包含車頭
        # ----------------------------------
        kernel = np.ones((15, 15), np.uint8)
        car_mask = cv2.dilate(car_mask, kernel, iterations=1)

        # =================================
        # 只留下車子附近的藍色
        # ==================================
        blue_car_mask = cv2.bitwise_and(blue_mask, car_mask)

        # ----------------------------------
        # 去除小雜訊
        # ----------------------------------
        clean_kernel = np.ones((3, 3), np.uint8)
        blue_car_mask = cv2.morphologyEx(blue_car_mask, cv2.MORPH_OPEN, clean_kernel)
        blue_car_mask = cv2.morphologyEx(blue_car_mask, cv2.MORPH_CLOSE, clean_kernel)

        # ==================================
        # 找藍色 contour
        # ==================================
        contours, _ = cv2.findContours(
            blue_car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, blue_car_mask

        # ----------------------------------
        # 最大藍色區域
        # ----------------------------------
        blue_contour = max(contours, key=cv2.contourArea)
        blue_area = cv2.contourArea(blue_contour)

        # 太小視為雜訊
        if blue_area < 5:
            return None, blue_car_mask

        # ==================================
        # Moment 找藍色區域中心
        # ==================================
        M = cv2.moments(blue_contour)
        if M["m00"] == 0:
            return None, blue_car_mask

        blue_x = int(M["m10"] / M["m00"])
        blue_y = int(M["m01"] / M["m00"])

        blue_center = (blue_x, blue_y)
        return (blue_center, blue_car_mask)

    # ======================================
    # ★ 利用藍色車頭計算 Heading
    # ======================================
    def heading_from_blue(self, car_center_image, blue_center):
        cx = car_center_image[0]
        cy = car_center_image[1]
        bx = blue_center[0]
        by = blue_center[1]

        # ==================================
        # 不直接使用 image dx/dy
        # 將車中心與藍色中心都轉成真實世界座標
        # 這樣 Heading 與 UDP 的 X/Y 座標系統一致
        # ==================================
        car_real = image_to_real(cx, cy, H_matrix)
        blue_real = image_to_real(bx, by, H_matrix)

        dx = blue_real[0] - car_real[0]
        dy = blue_real[1] - car_real[1]
        heading = math.degrees(math.atan2(dy, dx)) % 360.0
        return heading

    def detect_heading(self, frame, rect, car_center_real):
        # ==================================
        # 3. 車身長軸
        # ==================================
        body_angle = self.get_body_angle(rect)
        # ==================================
        # 4. ★ 找藍色車頭
        # ==================================
        blue_center, blue_mask = self.detect_blue_front(frame, rect)
        # ==================================
        # 5. 決定 Heading
        # minAreaRect 決定車身長軸，藍色 marker 決定哪一端是車頭
        # ==================================
        if blue_center is not None:
            heading1 = body_angle
            heading2 = (body_angle + 180.0) % 360.0

            blue_real = image_to_real(blue_center[0], blue_center[1], H_matrix)
            marker_dx = blue_real[0] - car_center_real[0]
            marker_dy = blue_real[1] - car_center_real[1]
            marker_angle = math.degrees(math.atan2(marker_dy, marker_dx)) % 360.0

            diff1 = abs(self.normalise_angle_change(heading1 - marker_angle))
            diff2 = abs(self.normalise_angle_change(heading2 - marker_angle))
            return heading1 if diff1 <= diff2 else heading2
        else:
            # 找不到藍色 marker 時才使用移動方向 fallback
            return self.determine_heading_from_motion(
                body_angle, car_center_real
            )


    def car_detected(self, frame, current_time_us, contour):
        # 預設為未偵測到賽車 (-1000.0, -1000.0)
        car_detected = False
        car_name = "unrecognised"

        # 取得最小外接矩形,朝向角度 (car_heading, degrees)
        rect = cv2.minAreaRect(contour)
        car_center_image, (width, height), car_heading_from_motion = rect
        car_center_image_int = [int(car_center_image[0]), int(car_center_image[1])]

        # 計算真實座標 (mm)
        car_center_real = image_coords_to_real(car_center_image, H_matrix)

        car_heading = self.detect_heading(frame, rect, car_center_real)

        if self.prev_time_us is None:
            self.prev_car_center_real = car_center_real
            self.prev_car_heading = car_heading
            self.prev_time_us = current_time_us
            return "no data yet\n"

        if current_time_us == self.prev_time_us:
            raise RuntimeError('Called twice on same frame')

        # 計算速度 velocity_smooth 與 角速度 (angular_velocity_smooth)
        dt = (current_time_us - self.prev_time_us) / 1000000.0  # 轉為秒

        # 速度 = 位置變化 / 時間 (mm/s)
        raw_diff = np.divide(np.subtract(car_center_real, self.prev_car_center_real), dt)

        # 修正角度跨越 +/-180 度問題
        d_theta = self.normalise_angle_change(car_heading - self.prev_car_heading)
        raw_omega = d_theta / dt

        # 一階指數平滑化
        self.velocity_smooth = self.smooth_vector(raw_diff, self.velocity_smooth)
        self.angular_velocity_smooth = self.smooth_scalar(raw_omega, self.angular_velocity_smooth)

        # 更新上一影格紀錄
        self.prev_car_center_real = car_center_real
        self.prev_car_heading = car_heading
        self.prev_time_us = current_time_us

        # 繪製車子框線與資訊
        self.annotate_image(frame, rect, car_center_image_int, self.car_name, car_heading, None)

        # 封裝輸出資訊字串 (符合老師要求格式)
        # 格式: timestamp:"car_id",x,y,car_heading,car_center_x_diff,car_center_y_diff,angular_velocity_smooth,u,w\n
        return (
            f'{current_time_us}:"{self.car_name}",'
            f'{car_center_real[0]:.1f},{car_center_real[1]:.1f},{car_heading:.1f},'
            f'{self.velocity_smooth[0]:.1f},{self.velocity_smooth[1]:.1f},{self.angular_velocity_smooth:.1f},'
            f'{car_center_image_int[0]},{car_center_image_int[1]}\n'
        )

# ==========================================
# 4. 賽車追蹤與狀態計算類別
# ==========================================
class ToyCarTracker:
    def __init__(self):
        self.previous_frame = None
        self.blur_kernel = np.ones((40, 40), np.float32) / 1600.0
        self.car_data = {}
        self.histogram_frames = []

    # Recognise a car:
    # 1. Extract the hue values of the pixels in the detected car area
    # 2. Compute an 8-way histogram of hue values
    # 3. Compare by vector distance against hue histograms from training data of various cars
    # 4. Take the best match
    def recognise_car(self, frame, mask):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        saturation = cv2.divide(hsv_frame[:,:,1], 16)
        lightness = cv2.divide(hsv_frame[:,:,2], 16)
        color_signal = cv2.multiply(saturation, lightness)
        max_signal = cv2.minMaxLoc(color_signal)[1]
        _, thresholded = cv2.threshold(color_signal, max_signal / 2, 255, cv2.THRESH_BINARY)

        # masked = cv2.bitwise_and(frame, frame, mask=thresholded)
        # cv2.imshow("histogram region", masked)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     exit()

        frame_histogram = cv2.calcHist([hsv_frame], [1], thresholded, [8], [0,256])
        frame_histogram_norm = np.linalg.norm(frame_histogram)
        if frame_histogram_norm == 0:
            return None
        frame_histogram_normalised = frame_histogram / frame_histogram_norm
        frame_histogram_normalised = [round(float(x[0]), 2) for x in frame_histogram_normalised]
        self.histogram_frames.append(frame_histogram_normalised)
        distances = [[car_name, np.linalg.norm(np.subtract(frame_histogram_normalised, car_histogram))]
                     for car_name, car_histogram in car_hue_histograms.items()]
        best_match = min(distances, key=lambda x:x[1])
        return best_match[0]

    def process_frame(self, frame, current_time_us):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.previous_frame is None:
            self.previous_frame = gray_frame
            return frame, []

        # 1. 運動影格相減 (Frame Differencing)
        difference = cv2.absdiff(gray_frame, self.previous_frame)
        self.previous_frame = gray_frame

        blurred = cv2.filter2D(difference, -1, self.blur_kernel)
        max_diff = cv2.minMaxLoc(blurred)[1]

        udp_output_string = ""

        if max_diff < 10:
            return frame, []

        ret, thresholded = cv2.threshold(blurred, max_diff / 2, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours is None:
            return frame, []

        udp_output_strings = []
        detected_cars = {}

        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue

            (centre_x, centre_y, width, height) = cv2.boundingRect(contour)
            car_region = frame[centre_y:centre_y + height, centre_x:centre_x + width]
            mask_region = thresholded[centre_y:centre_y + height, centre_x:centre_x + width]
            car_name = self.recognise_car(car_region, mask_region)
            if car_name is None:
                continue

            if car_name in detected_cars:
                continue
            detected_cars[car_name] = True

            if car_name not in self.car_data:
                self.car_data[car_name] = CarData()
            car_data = self.car_data[car_name]
            car_data.car_name = car_name

            udp_output_strings.append(car_data.car_detected(frame, current_time_us, contour))

        return frame, udp_output_strings


def handle_frame(frame):
    # 取得微秒時間戳記 (microseconds relative clock)
    current_time_us = int((time.time() - start_time) * 1000000)

    # 處理影格
    processed_frame, output_messages = tracker.process_frame(frame, current_time_us)

    for output_message in output_messages:
        # 1. 終端機即時印出訊息
        print(output_message, end='')

        # 2. UTF-8 編碼並透過 UDP 發送至 Port 5000
        sock.sendto(output_message.encode('utf-8'), (UDP_IP, UDP_PORT))

    # 顯示即時畫面
    resized = resize_frame(processed_frame, 640, 640)
    cv2.imshow('Global Vision Server', resized)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        return None

    return frame

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("錯誤：無法開啟影片。")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
video_output = cv2.VideoWriter("output.mp4", fourcc, frame_rate, (frame_width, frame_height))

tracker = ToyCarTracker()
start_time = time.time()

# ==========================================
# 5. 主程式迴圈
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("影片播放完畢。")
        break

    frame = handle_frame(frame)
    if frame is None:
        break

    video_output.write(frame)

average_histogram = np.average(tracker.histogram_frames, axis=0)
print(average_histogram)

cap.release()
video_output.release()

cv2.destroyAllWindows()
sock.close()
