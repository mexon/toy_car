import cv2
import numpy as np
import os
import time
import socket
import math
import sys

# ==========================================
# 1. UDP 網路設定
# ==========================================
UDP_IP = "127.0.0.1"
UDP_PORT = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ==========================================
# 2. 影片路徑
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = sys.argv[1]

if not os.path.exists(video_path):
    print(f"錯誤：找不到檔案 '{video_path}'")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("錯誤：無法開啟影片")
    sys.exit()

# ==========================================
# 3. Homography
# Pixel -> 真實世界座標
# ==========================================

# 注意：
# 這四個點之後要換成你影片真正的場地角點

pts_image = np.array(
    [[100, 100], [1180, 100], [1180, 620], [100, 620]], dtype=np.float32
)

# 真實場地大小
# 2500 mm x 1500 mm
pts_real = np.array(
    [[0.0, 0.0], [2500.0, 0.0], [2500.0, 1500.0], [0.0, 1500.0]], dtype=np.float32
)
H_matrix, _ = cv2.findHomography(pts_image, pts_real)

# ==========================================
# 4. 車輛 Histogram
# ==========================================
car_hue_histograms = {
    "car1": [0.0, 0.21, 0.51, 0.53, 0.44, 0.3, 0.15, 0.18],
    "car2": [0.05, 0.07, 0.09, 0.15, 0.38, 0.78, 0.29, 0.31],
}

# ==========================================
# 5. Pixel -> Real World
# ==========================================
def image_to_real(u, v, H):
    pt = np.array([u, v, 1.0], dtype=np.float32).reshape(3, 1)
    real_pt = np.dot(H, pt)
    real_pt /= real_pt[2]
    return (float(real_pt[0][0]), float(real_pt[1][0]))

def image_coords_to_real(coords, H):
    return image_to_real(coords[0], coords[1], H)

# ==========================================
# 6. CarData
# ==========================================
class CarData:
    alpha = 0.3

    def __init__(self):
        self.car_name = "unrecognised"
        # 上一幀位置

        self.prev_car_center_real = None

        # 上一幀 Heading

        self.prev_car_orientation = None
        # 上一幀時間

        self.prev_time_us = None

        # 平滑速度

        self.v_smooth = [0.0, 0.0]

        # 平滑角速度
        self.angular_velocity_smooth = 0.0

    # ======================================
    # 平滑
    # ======================================
    def smooth_scalar(self, new_value, old_value):

        return self.alpha * new_value + (1.0 - self.alpha) * old_value

    def smooth_vector(self, new_value, old_value):

        return [self.smooth_scalar(new, old) for new, old in zip(new_value, old_value)]

    # ======================================
    # 最短角度差
    # ======================================
    def angle_difference(self, new_angle, old_angle):
        return (new_angle - old_angle + 180.0) % 360.0 - 180.0

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
            if self.prev_car_orientation is not None:
                return self.prev_car_orientation
            return body_angle

        dx = car_center_real[0] - self.prev_car_center_real[0]
        dy = car_center_real[1] - self.prev_car_center_real[1]

        distance = math.hypot(dx, dy)
        # 幾乎沒動

        if distance < 5.0:
            if self.prev_car_orientation is not None:
                return self.prev_car_orientation
            return body_angle

        motion_angle = math.degrees(math.atan2(dy, dx)) % 360.0
        heading1 = body_angle
        heading2 = (body_angle + 180.0) % 360.0
        diff1 = abs(self.angle_difference(heading1, motion_angle))
        diff2 = abs(self.angle_difference(heading2, motion_angle))
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

    # ======================================
    # 畫追蹤結果
    # ======================================
    def annotate_image(
        self, frame, rect, car_center_image_int, car_name, heading, blue_center
    ):
        # 綠色：車身框
        box = np.int64(cv2.boxPoints(rect))
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

        cx, cy = car_center_image_int
        # 紅色：車身中心
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"ID: {car_name}",
            (cx + 10, cy - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Heading: {heading:.1f}",
            (cx + 10, cy + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
        )

        # 藍色：偵測到的車頭 marker
        if blue_center is not None:
            cv2.circle(frame, blue_center, 8, (255, 0, 0), 2)

        # 紫色箭頭：最終 Heading
        arrow_length = 80
        theta = math.radians(heading)
        end_x = int(cx + arrow_length * math.cos(theta))
        end_y = int(cy + arrow_length * math.sin(theta))
        cv2.arrowedLine(
            frame, (cx, cy), (end_x, end_y), (255, 0, 255), 3, tipLength=0.3
        )

    def car_detected(self, frame, current_time_us, contour):
        # ==================================
        # 1. minAreaRect
        # ==================================
        rect = cv2.minAreaRect(contour)
        car_center_image, (width, height), angle = rect
        car_center_image_int = [int(car_center_image[0]), int(car_center_image[1])]
        # ==================================
        # 2. 真實世界位置
        # ==================================
        car_center_real = image_coords_to_real(car_center_image, H_matrix)
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

            diff1 = abs(self.angle_difference(heading1, marker_angle))
            diff2 = abs(self.angle_difference(heading2, marker_angle))
            car_orientation = heading1 if diff1 <= diff2 else heading2
        else:
            # 找不到藍色 marker 時才使用移動方向 fallback
            car_orientation = self.determine_heading_from_motion(
                body_angle, car_center_real
            )

        # ==================================
        # 6. 計算速度與角速度
        # ==================================

        if (
            self.prev_time_us is not None
            and self.prev_car_center_real is not None
            and self.prev_car_orientation is not None
        ):

            dt = (current_time_us - self.prev_time_us) / 1000000.0

            if dt > 0:
                # --------------------------
                # X/Y 位移
                # --------------------------
                dx = car_center_real[0] - self.prev_car_center_real[0]
                dy = car_center_real[1] - self.prev_car_center_real[1]
                # --------------------------
                # 速度 mm/s
                # --------------------------

                raw_velocity = [dx / dt, dy / dt]

                self.v_smooth = self.smooth_vector(raw_velocity, self.v_smooth)
                # --------------------------
                # Heading 變化
                # --------------------------
                d_theta = self.angle_difference(
                    car_orientation, self.prev_car_orientation
                )
                # --------------------------
                # 角速度 deg/s
                # --------------------------
                raw_omega = d_theta / dt
                self.angular_velocity_smooth = self.smooth_scalar(
                    raw_omega, self.angular_velocity_smooth
                )
        # ==================================
        # 7. 畫結果
        # ==================================
        self.annotate_image(
            frame,
            rect,
            car_center_image_int,
            self.car_name,
            car_orientation,
            blue_center,
        )
        # ==================================
        # 8. 更新上一幀
        # ==================================
        self.prev_car_center_real = car_center_real
        self.prev_car_orientation = car_orientation
        self.prev_time_us = current_time_us
        # ==================================
        # 9. UDP
        # ==================================
        return (
            f"{current_time_us}:"
            f'"{self.car_name}",'
            f"{car_center_real[0]:.1f},"
            f"{car_center_real[1]:.1f},"
            f"{car_orientation:.1f},"
            f"{self.v_smooth[0]:.1f},"
            f"{self.v_smooth[1]:.1f},"
            f"{self.angular_velocity_smooth:.1f},"
            f"{car_center_image_int[0]},"
            f"{car_center_image_int[1]}\n"
        )


# ==========================================
# 7. ToyCarTracker
# ==========================================
class ToyCarTracker:
    def __init__(self):
        self.previous_frame = None
        self.blur_kernel = np.ones((40, 40), np.float32) / 1600.0
        self.car_data = CarData()

    # ======================================
    # 車輛 ID 辨識
    # ======================================
    def recognise_car(self, frame, mask):

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 目前先維持你原本的
        # Saturation histogram
        frame_histogram = cv2.calcHist([hsv_frame], [1], mask, [8], [0, 256])

        histogram_norm = np.linalg.norm(frame_histogram)
        if histogram_norm == 0:
            return "unrecognised"

        frame_histogram_normalised = frame_histogram / histogram_norm
        distances = [
            [car_name, np.linalg.norm(frame_histogram_normalised - car_histogram)]
            for car_name, car_histogram in car_hue_histograms.items()
        ]
        # distance 越小越像

        best_match = min(distances, key=lambda x: x[1])
        return best_match[0]

    # ======================================
    # Frame Processing
    # ======================================
    def process_frame(self, frame, current_time_us):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 第一幀

        if self.previous_frame is None:
            self.previous_frame = gray_frame
            return frame, None

        # ==================================
        # Frame Differencing
        # ==================================
        difference = cv2.absdiff(gray_frame, self.previous_frame)

        self.previous_frame = gray_frame
        # ==================================
        # Blur
        # =================================
        blurred = cv2.filter2D(difference, -1, self.blur_kernel)
        max_diff = cv2.minMaxLoc(blurred)[1]
        udp_output_string = ""
        # ==================================
        # 有明顯運動
        # ==================================
        if max_diff >= 10:
            # ------------------------------
            # Threshold
            # ------------------------------
            ret, thresholded = cv2.threshold(
                blurred, max_diff / 2, 255, cv2.THRESH_BINARY
            )

            # ------------------------------
            # Contours
            # ------------------------------
            contours, _ = cv2.findContours(
                thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:

                # 最大移動區域
                cnt = max(contours, key=cv2.contourArea)
                # 過濾小雜訊
                if cv2.contourArea(cnt) > 100:
                    # ----------------------
                    # ID
                    # ----------------------
                    self.car_data.car_name = self.recognise_car(frame, thresholded)
                    # ----------------------
                    # Position / Heading
                    # ----------------------
                    udp_output_string = self.car_data.car_detected(
                        frame, current_time_us, cnt
                    )
        return (frame, udp_output_string)


# ==========================================
# 8. 主程式
# ==========================================

# ==========================================
# 點擊畫面取得 HSV
# ==========================================

tracker = ToyCarTracker()

start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("影片播放完畢。")
        break
    current_time_us = int((time.time() - start_time) * 1000000)

    processed_frame, output_msg = tracker.process_frame(frame, current_time_us)
    if output_msg:
        print(output_msg, end="")
        sock.sendto(output_msg.encode("utf-8"), (UDP_IP, UDP_PORT))
    display_frame = cv2.resize(processed_frame, None, fx=0.7, fy=0.7)
    cv2.imshow("Global Vision Server", display_frame)
    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
