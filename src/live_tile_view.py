"""
live_tile_view.py
Continuously monitor a live camera (or a video file played like a live feed),
detect the floor-tile grid on every frame and show it in a window.

    python live_tile_view.py --source 0                 # webcam / robot camera
    python live_tile_view.py --source test_video.mp4    # replay a video as "live"
    python live_tile_view.py --source test_video.mp4 --loop --every 2

Keys:  q = quit    space = pause    s = save calibration JSON
       l = LOCK / UNLOCK calibration (fixed camera):
           locked  -> stops detecting, draws the 60 cm grid projected from the saved
                      homography, so you can see if the calibration still lines up
                      with the real tiles while robots/people move around.

Needs tile_calibration.py in the same folder.
"""
import argparse, json, time
import cv2
import numpy as np
import tile_calibration as tc

ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
       270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def draw_locked_grid(frame, H, tile=60.0, nx=6, ny=8, x0=-60, y0=-60):
    """Project a real-world grid (cm) back into the image using the locked H."""
    Hinv = np.linalg.inv(H)
    out = frame.copy()
    h, w = frame.shape[:2]

    def proj(p):
        q = cv2.perspectiveTransform(np.array([[p]], np.float32), Hinv)[0, 0]
        return int(q[0]), int(q[1])

    for i in range(nx + 1):
        x = x0 + i * tile
        a, b = proj([x, y0]), proj([x, y0 + ny * tile])
        cv2.line(out, a, b, (0, 255, 255), 1)
    for j in range(ny + 1):
        y = y0 + j * tile
        a, b = proj([x0, y]), proj([x0 + nx * tile, y])
        cv2.line(out, a, b, (255, 255, 0), 1)
    return out


def hud(img, lines, color=(0, 255, 0)):
    for i, t in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="camera index or video file")
    ap.add_argument("--every", type=int, default=1, help="run detection every Nth frame")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--loop", action="store_true", help="restart a video file when it ends")
    ap.add_argument("--stale", type=int, default=15,
                    help="keep showing the last good detection this many frames after a failure")
    ap.add_argument("--no-display", action="store_true", help="headless (no window)")
    ap.add_argument("--save-video", default="", help="write annotated output to this .mp4")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    ap.add_argument("--json", default="live_calibration.json")
    args = ap.parse_args()

    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")

    writer = None
    last_good, last_good_frame_i = None, -10**9
    locked_H = None
    paused, n, t_prev, fps = False, 0, time.time(), 0.0
    n_ok = n_fail = 0
    fail_msg, last_attempt_ok = "", False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                if args.loop and not is_cam:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            if ROT[args.rotate] is not None:
                frame = cv2.rotate(frame, ROT[args.rotate])
            n += 1

            if locked_H is not None:
                view = draw_locked_grid(frame, locked_H)
                status = ["LOCKED: projected 60 cm grid from saved calibration",
                          "grid should stay on the tile lines (press l to re-detect)"]
                color = (0, 200, 255)
            else:
                if n % args.every == 0 or last_good is None:
                    cal = tc.calibrate_from_frame(frame)
                    if cal is not None:
                        last_good, last_good_frame_i = cal, n
                        n_ok += 1
                        last_attempt_ok = True
                    else:
                        n_fail += 1
                        fail_msg = tc.LAST_FAILURE
                        last_attempt_ok = False
                age = n - last_good_frame_i
                if last_good is not None and last_attempt_ok:
                    view = last_good.debug_frame.copy()
                    status = [f"DETECTING  points={len(last_good.pts_image)}  "
                              f"error={last_good.rms_cm:.2f} cm"]
                    color = (0, 255, 0)
                elif last_good is not None and age <= args.stale:
                    # keep showing the recent result but mark it as stale
                    view = frame.copy()
                    status = [f"NO GRID this frame ({fail_msg}); showing last good {age} frames ago"]
                    color = (0, 165, 255)
                else:
                    view = frame.copy()
                    status = [f"NO GRID: {fail_msg if n_fail else 'searching...'}"]
                    color = (0, 0, 255)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - t_prev)) if fps else 1.0 / max(1e-6, now - t_prev)
            t_prev = now
            hud(view, status + [f"{fps:4.1f} FPS   ok={n_ok} fail={n_fail}", "q quit | l lock | s save | space pause"], color)

            if args.save_video:
                if writer is None:
                    h, w = view.shape[:2]
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                writer.write(view)
            if args.max_frames and n >= args.max_frames:
                break

        if not args.no_display:
            shown = view
            if view.shape[1] > 1100 or view.shape[0] > 800:   # fit window to a normal screen
                k = min(1100 / view.shape[1], 800 / view.shape[0])
                shown = cv2.resize(view, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            cv2.imshow("Live tile grid (q quit)", shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("l"):
                if locked_H is None and last_good is not None:
                    locked_H = last_good.H.copy()
                    print("[INFO] calibration LOCKED")
                else:
                    locked_H = None
                    print("[INFO] calibration unlocked, detecting again")
            elif key == ord("s") and last_good is not None:
                json.dump({"pts_image": last_good.pts_image.tolist(),
                           "pts_real": last_good.pts_real.tolist(),
                           "H": last_good.H.tolist()}, open(args.json, "w"), indent=2)
                print(f"[INFO] saved {args.json}")

    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    total = n_ok + n_fail
    if total:
        print(f"Processed {n} frames; detection ran {total} times, ok {n_ok} ({100*n_ok/total:.0f}%), "
              f"average {fps:.1f} FPS")


if __name__ == "__main__":
    main()
