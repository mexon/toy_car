"""
===============================================================================
Floor Tile Corner & Edge Detection for Camera Calibration
===============================================================================
This module detects floor tile grid line intersections (corners) from an image 
or real-time camera stream, filters out non-floor background noise, labels corners 
relative to the bottom-left of the playing field, and outputs coordinates to a 
JSON file and copy-pasteable NumPy arrays.

Key Features:
- Supports both static images and live webcam streams (cv2.VideoCapture).
- Black Top-Hat morphological filtering for robust dark grid line enhancement.
- Probabilistic Hough Line Transform & Total Least Squares line fitting.
- HSV color & geometric ROI masking to filter background noise (furniture/cables).
- Sub-pixel corner refinement (cv2.cornerSubPix).
- Grid labeling ordered relative to LEFT (1st, 2nd...) and BOTTOM (1st, 2nd...).
- High-contrast, large-font visual labels with background text boxes for clarity.
- Exports coordinates to a structured JSON file and calculates Homography (H).

Author: Antigravity Team
===============================================================================
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np


def ordinal(n: int) -> str:
    """Converts an integer to its ordinal string (e.g., 1 -> '1st', 2 -> '2nd')."""
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def line_intersection(line1: np.ndarray, line2: np.ndarray) -> np.ndarray:
    """
    Computes the 2D intersection point (x, y) of two lines defined by (A, B, C) 
    where Ax + By + C = 0. Returns None if lines are parallel.
    """
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-6:
        return None
    x = (B1 * C2 - B2 * C1) / det
    y = (A2 * C1 - A1 * C2) / det
    return np.array([x, y], dtype=np.float32)


def fit_line_from_points(pts: list) -> np.ndarray:
    """
    Fits a normalized line Ax + By + C = 0 to a set of 2D points using 
    Total Least Squares (cv2.fitLine).
    """
    pts_arr = np.array(pts, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    A = -vy
    B = vx
    C = vy * x0 - vx * y0
    norm = np.sqrt(A**2 + B**2)
    return np.array([A / norm, B / norm, C / norm])


def cluster_by_gap(items: list, max_gap: float = 25.0) -> list:
    """
    Clusters 1D items (sorted by position) if the gap between consecutive items 
    is less than max_gap.
    """
    if not items:
        return []
    sorted_items = sorted(items, key=lambda x: x[0])
    clusters = [[sorted_items[0]]]
    for item in sorted_items[1:]:
        if item[0] - clusters[-1][-1][0] <= max_gap:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def create_playing_field_mask(image: np.ndarray) -> np.ndarray:
    """
    Generates a binary mask of the valid floor playing field by masking out 
    non-floor objects (e.g., red stool, white cabinet wall, power cables).
    """
    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # 1. HSV Red color masking (filters out red stool)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    mask_red = cv2.dilate(mask_red1 | mask_red2, 
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), 
                          iterations=2)
    mask[mask_red > 0] = 0

    # 2. Geometric masking for top-right cabinet wall
    pts_cabinet = np.array([[int(w * 0.52), 0], [w, 0], [w, int(h * 0.24)], [int(w * 0.75), int(h * 0.24)]], np.int32)
    cv2.fillPoly(mask, [pts_cabinet], 0)

    # 3. Geometric masking for far-right wall cables
    pts_cables = np.array([[int(w * 0.88), int(h * 0.39)], [w, int(h * 0.35)], [w, h], [int(w * 0.88), h]], np.int32)
    cv2.fillPoly(mask, [pts_cables], 0)

    return mask


def draw_high_contrast_label(img: np.ndarray, text: str, pos: tuple, text_color=(0, 255, 255), bg_color=(0, 0, 0)):
    """
    Draws text with a solid high-contrast background bounding box for maximum readability.
    """
    x, y = pos
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    thickness = 2
    
    # Get text width & height
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Bounding box coordinates
    margin = 4
    box_x1 = x
    box_y1 = y - text_h - margin
    box_x2 = x + text_w + (margin * 2)
    box_y2 = y + baseline + margin
    
    # Keep inside image bounds
    h, w = img.shape[:2]
    if box_x2 > w:
        box_x1 = max(0, x - text_w - margin * 2)
        box_x2 = min(w, x)
    if box_y1 < 0:
        box_y1 = y + 8
        box_y2 = y + 8 + text_h + margin * 2
        
    # Draw filled background rectangle
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
    # Draw bright green border around rectangle
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 1)
    
    # Draw text
    text_pos_y = box_y1 + text_h + margin - 1
    cv2.putText(img, text, (box_x1 + margin, text_pos_y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def process_frame(frame: np.ndarray, tile_size_cm: float = 60.0, crop_offset: tuple = (0, 0)):
    """
    Processes a single BGR image frame to detect tile grid intersections, 
    filter noise, index grid corners, and refine sub-pixel positions.

    Parameters:
    - frame: BGR input image frame
    - tile_size_cm: Real world width/height of a tile in cm (default 60.0 cm)
    - crop_offset: (offset_x, offset_y) relative to full uncropped frame

    Returns:
    - intersections: List of dicts containing grid positions, pixel coords, and real coords.
    - vis_frame: Annotated output image with grid lines and labeled corners.
    - H: Homography matrix (pts_image -> pts_real)
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Create playing field mask (exclude non-tile objects)
    floor_mask = create_playing_field_mask(frame)

    # 2. Morphological Black Top-Hat transform to enhance dark grid lines on bright floor
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blurred = cv2.GaussianBlur(tophat, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)

    # 3. Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=30)
    if lines is None:
        return [], frame, None

    v_items = []
    h_items = []
    ref_y, ref_x = h / 2.0, w / 2.0

    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        p1, p2 = np.array([x1, y1]), np.array([x2, y2])

        if 65 <= angle <= 115: # Vertical grid lines
            x_mid = x1 + (ref_y - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1
            v_items.append((x_mid, p1, p2))
        elif angle <= 30 or angle >= 150: # Horizontal grid lines
            y_mid = y1 + (ref_x - x1) * (y2 - y1) / (x2 - x1) if x2 != x1 else y1
            h_items.append((y_mid, p1, p2))

    # 4. Cluster line segments into major grid lines
    v_clusters = cluster_by_gap(v_items, max_gap=25.0)
    h_clusters = cluster_by_gap(h_items, max_gap=25.0)

    # Fit consensus line equations
    v_lines = []
    for cluster in v_clusters:
        pts = [p for item in cluster for p in (item[1], item[2])]
        line_eq = fit_line_from_points(pts)
        A, B, C = line_eq
        x_at_mid = -(B * ref_y + C) / A
        v_lines.append((x_at_mid, line_eq))
    v_lines.sort(key=lambda x: x[0]) # Left to Right: Col 1, Col 2, Col 3...

    h_lines = []
    for cluster in h_clusters:
        pts = [p for item in cluster for p in (item[1], item[2])]
        line_eq = fit_line_from_points(pts)
        A, B, C = line_eq
        y_at_mid = -(A * ref_x + C) / B
        h_lines.append((y_at_mid, line_eq))
    h_lines.sort(key=lambda x: x[0])
    
    # Reverse horizontal lines so index 0 is 1st from BOTTOM!
    h_lines_from_bottom = list(reversed(h_lines))

    # 5. Compute Intersections & Filter Noise
    raw_intersections = []
    ox, oy = crop_offset

    for col_idx, (_, v_eq) in enumerate(v_lines):        # col_idx=0 -> 1st from left
        for row_idx, (_, h_eq) in enumerate(h_lines_from_bottom): # row_idx=0 -> 1st from bottom
            pt = line_intersection(v_eq, h_eq)
            if pt is not None:
                px, py = pt[0], pt[1]
                if 0 <= px < w and 0 <= py < h:
                    ix, iy = int(round(px)), int(round(py))
                    # Validate point falls within clean playing field floor
                    if floor_mask[iy, ix] > 0:
                        raw_intersections.append({
                            'col_idx': col_idx + 1,
                            'row_idx': row_idx + 1,
                            'col_label': f"{ordinal(col_idx + 1)} from left",
                            'row_label': f"{ordinal(row_idx + 1)} from bottom",
                            'short_label': f"({col_idx+1}L, {row_idx+1}B)",
                            'pt_img': np.array([px, py], dtype=np.float32),
                            'pt_img_uncropped': np.array([px + ox, py + oy], dtype=np.float32),
                            'pt_real': np.array([col_idx * tile_size_cm, row_idx * tile_size_cm], dtype=np.float32)
                        })

    if not raw_intersections:
        return [], frame, None

    # 6. Sub-pixel Corner Refinement
    pts_raw = np.array([item['pt_img'] for item in raw_intersections], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pts_refined = cv2.cornerSubPix(gray, pts_raw, (7, 7), (-1, -1), criteria)

    intersections = []
    for idx, (item, pt_ref) in enumerate(zip(raw_intersections, pts_refined)):
        item['pt_img'] = pt_ref
        item['pt_img_uncropped'] = pt_ref + np.array([ox, oy], dtype=np.float32)
        item['id'] = f"P{idx+1:02d}"
        intersections.append(item)

    # 7. Calculate Homography Matrix H (pts_image -> pts_real)
    pts_img_arr = np.array([item['pt_img'] for item in intersections], dtype=np.float32)
    pts_real_arr = np.array([item['pt_real'] for item in intersections], dtype=np.float32)
    
    H, _ = cv2.findHomography(pts_img_arr, pts_real_arr, cv2.RANSAC, 5.0)

    # 8. Annotate Output Frame with High Clarity Labels
    vis_frame = frame.copy()
    
    # Draw vertical grid lines (yellow)
    for _, (A, B, C) in v_lines:
        cv2.line(vis_frame, (int(round(-(B*0+C)/A)), 0), (int(round(-(B*h+C)/A)), h), (0, 255, 255), 2)
        
    # Draw horizontal grid lines (cyan)
    for _, (A, B, C) in h_lines_from_bottom:
        cv2.line(vis_frame, (0, int(round(-(A*0+C)/B))), (w, int(round(-(A*w+C)/B))), (255, 255, 0), 2)

    # Draw intersection targets & high-contrast text boxes
    for item in intersections:
        px, py = item['pt_img']
        pt_int = (int(round(px)), int(round(py)))
        
        # Red outer circle + Green inner dot + Crosshair lines
        cv2.circle(vis_frame, pt_int, 7, (0, 255, 0), 2)
        cv2.circle(vis_frame, pt_int, 3, (0, 0, 255), -1)
        
        # Crosshairs
        cv2.line(vis_frame, (pt_int[0] - 10, pt_int[1]), (pt_int[0] + 10, pt_int[1]), (0, 255, 0), 1)
        cv2.line(vis_frame, (pt_int[0], pt_int[1] - 10), (pt_int[0], pt_int[1] + 10), (0, 255, 0), 1)
        
        # High contrast background label box
        lbl_text = f"{item['id']}:{item['short_label']}"
        draw_high_contrast_label(vis_frame, lbl_text, (pt_int[0] + 8, pt_int[1] - 5))

    return intersections, vis_frame, H


def save_coordinates_to_json(intersections: list, H: np.ndarray, json_path: str = 'tile_corners_coordinates.json', tile_size_cm: float = 60.0):
    """
    Exports all detected corner coordinates, grid positions, and calibration 
    matrices to a clean JSON file.
    """
    corners_data = []
    pts_image_list = []
    pts_real_list = []

    for item in intersections:
        corner_entry = {
            "id": item['id'],
            "col_from_left": int(item['col_idx']),
            "row_from_bottom": int(item['row_idx']),
            "grid_label": f"{item['col_label']}, {item['row_label']}",
            "short_label": item['short_label'],
            "pixel_coordinate_floor_crop": [round(float(item['pt_img'][0]), 2), round(float(item['pt_img'][1]), 2)],
            "pixel_coordinate_full_image": [round(float(item['pt_img_uncropped'][0]), 2), round(float(item['pt_img_uncropped'][1]), 2)],
            "real_world_coordinate_cm": [round(float(item['pt_real'][0]), 1), round(float(item['pt_real'][1]), 1)]
        }
        corners_data.append(corner_entry)
        pts_image_list.append(corner_entry['pixel_coordinate_floor_crop'])
        pts_real_list.append(corner_entry['real_world_coordinate_cm'])

    json_output = {
        "metadata": {
            "description": "Floor Tile Corner Coordinates for Camera Calibration",
            "tile_size_cm": tile_size_cm,
            "total_corners_detected": len(intersections),
            "col_indexing": "1st from left, 2nd from left, ...",
            "row_indexing": "1st from bottom, 2nd from bottom, ..."
        },
        "corners": corners_data,
        "pts_image": pts_image_list,
        "pts_real": pts_real_list,
        "homography_matrix_H": H.tolist() if H is not None else []
    }

    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=4)
        
    print(f"[INFO] Successfully saved coordinates to JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Floor Tile Edge & Corner Detector for Camera Calibration")
    parser.add_argument("--source", type=str, default="floor_cropped.jpg", 
                        help="Path to image file OR camera index (e.g., '0' for webcam stream)")
    parser.add_argument("--json", type=str, default="tile_corners_coordinates.json", 
                        help="Output JSON file path for detected coordinates")
    parser.add_argument("--tile-size", type=float, default=60.0, 
                        help="Real world width/height of floor tile in cm (default: 60.0)")
    parser.add_argument("--crop", action="store_true", 
                        help="Crop floor area if loading screenshot")
    
    args = parser.parse_args()

    # Determine if source is live camera feed or static image file
    is_live_camera = args.source.isdigit()

    if is_live_camera:
        cam_idx = int(args.source)
        print(f"[INFO] Opening live camera feed (Index {cam_idx})... Press 'q' to quit, 's' to save JSON.")
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {cam_idx}")
            sys.exit(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame from camera stream.")
                break

            intersections, vis_frame, H = process_frame(frame, tile_size_cm=args.tile_size)

            cv2.imshow("Live Tile Camera Calibration (Press 'q' to exit, 's' to save)", vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if intersections:
                    save_coordinates_to_json(intersections, H, args.json, args.tile_size)
                    cv2.imwrite("live_calibration_frame.png", vis_frame)
                    print("[INFO] Saved frame and JSON coordinates.")

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Static Image Processing Mode
        image_path = args.source
        if not os.path.exists(image_path):
            if os.path.exists('floor_cropped.jpg'):
                image_path = 'floor_cropped.jpg'
            elif os.path.exists('WhatsApp Image 2026-10-05 at 9.25.05 AM.jpeg'):
                image_path = 'WhatsApp Image 2026-10-05 at 9.25.05 AM.jpeg'
            else:
                print(f"[ERROR] File not found: {args.source}")
                sys.exit(1)

        print(f"[INFO] Processing static image file: {image_path}")
        img = cv2.imread(image_path)
        
        crop_offset = (0, 0)
        if "WhatsApp" in image_path or args.crop:
            img = img[525:1113, 100:538]
            crop_offset = (100, 525)

        intersections, vis_frame, H = process_frame(img, tile_size_cm=args.tile_size, crop_offset=crop_offset)

        if intersections:
            print(f"[SUCCESS] Detected {len(intersections)} playing field tile corners.")
            
            # Print coordinates to console
            print("\n" + "="*75)
            print("DETECTED CORNER COORDINATES (P01 to P17):")
            print("="*75)
            print(f"{'ID':<6} | {'Grid Position':<32} | {'Pixel (x, y)':<18} | {'Real (X, Y) cm':<14}")
            print("-" * 75)
            for item in intersections:
                px, py = item['pt_img']
                rx, ry = item['pt_real']
                grid_str = f"{item['col_label']}, {item['row_label']}"
                print(f"{item['id']:<6} | {grid_str:<32} | ({px:6.2f}, {py:6.2f}) | ({rx:5.1f}, {ry:5.1f})")
            print("="*75 + "\n")

            # Save JSON file
            save_coordinates_to_json(intersections, H, args.json, args.tile_size)
            
            # Save crisp visual outputs
            cv2.imwrite("grid_intersections_labeled.png", vis_frame)
            cv2.imwrite("tile_calibration_final_result.png", vis_frame)
            print("[INFO] Saved crisp visual overlay to 'grid_intersections_labeled.png' and 'tile_calibration_final_result.png'")

            # Annotate full WhatsApp image if available
            orig_img_path = 'WhatsApp Image 2026-10-05 at 9.25.05 AM.jpeg'
            if os.path.exists(orig_img_path):
                orig_img = cv2.imread(orig_img_path)
                vis_orig = orig_img.copy()
                for item in intersections:
                    px, py = item['pt_img_uncropped']
                    pt_int = (int(round(px)), int(round(py)))
                    cv2.circle(vis_orig, pt_int, 8, (0, 255, 0), 2)
                    cv2.circle(vis_orig, pt_int, 4, (0, 0, 255), -1)
                    lbl = f"{item['id']}:{item['short_label']}"
                    draw_high_contrast_label(vis_orig, lbl, (pt_int[0] + 10, pt_int[1] - 5))
                cv2.imwrite("whatsapp_intersections_labeled.png", vis_orig)

            if H is not None:
                print("\nCalculated Homography Matrix H (pts_image -> pts_real [cm]):")
                print(np.array2string(H, formatter={'float_kind': lambda x: f"{x:12.6f}"}))

        else:
            print("[WARNING] No tile grid intersections were detected.")


if __name__ == "__main__":
    main()
