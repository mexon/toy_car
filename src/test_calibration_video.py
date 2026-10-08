"""
test_calibration_video.py
Run tile calibration on a recorded video (or a live camera index) and report
how reliable it is, so you can test without access to the hardware.

Usage:
    python test_calibration_video.py --source taiwan_recording.mp4
    python test_calibration_video.py --source taiwan_recording.mp4 --every 15 --rotate 90
    python test_calibration_video.py --source 0            # live camera, later

Outputs (in --out, default calib_test_out/):
    summary.json        per-frame results + stability stats
    best_frame.png      debug overlay of the best frame (lowest error, most points)
    worst_frame.png     debug overlay of a failed/lowest-quality frame
    chosen_calibration.json   pts_image / pts_real / H from the best frame
"""
import argparse, json, os
import cv2
import numpy as np
import tile_calibration
from tile_calibration import calibrate_from_frame

ROTATE = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
          270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def project_real_to_image(H, real_pts):
    Hinv = np.linalg.inv(H)
    return cv2.perspectiveTransform(real_pts.reshape(-1, 1, 2).astype(np.float32), Hinv).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path or camera index")
    ap.add_argument("--every", type=int, default=10, help="test every Nth frame")
    ap.add_argument("--max-frames", type=int, default=300, help="max frames to test")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate frames if the video is sideways")
    ap.add_argument("--out", default="calib_test_out")
    ap.add_argument("--tile-size", type=float, default=60.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")

    print(f"OpenCV {cv2.__version__}; first frame size will be printed below")
    results, frame_i, tested = [], -1, 0
    fail_reasons = {}
    best, worst_frame = None, None
    all_H = []
    while tested < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if frame_i % args.every:
            continue
        if ROTATE[args.rotate] is not None:
            frame = cv2.rotate(frame, ROTATE[args.rotate])
        tested += 1
        if tested == 1:
            print(f"Frame size: {frame.shape[1]}x{frame.shape[0]} (width x height)")
        cal = calibrate_from_frame(frame, tile_size_cm=args.tile_size)
        if cal is None:
            why = tile_calibration.LAST_FAILURE.split(" (")[0].split(" v=")[0]
            fail_reasons[why] = fail_reasons.get(why, 0) + 1
            results.append({"frame": frame_i, "ok": False, "reason": tile_calibration.LAST_FAILURE})
            if worst_frame is None:
                worst_frame = frame.copy()
            continue
        all_H.append(cal.H)
        results.append({"frame": frame_i, "ok": True, "n_points": len(cal.pts_image),
                        "rms_cm": round(cal.rms_cm, 3)})
        # best = most points, then lowest error
        key = (len(cal.pts_image), -cal.rms_cm)
        if best is None or key > best[0]:
            best = (key, cal, frame_i)
    cap.release()

    ok_res = [r for r in results if r["ok"]]
    if fail_reasons:
        print("Failure reasons:", fail_reasons)
    print(f"Tested {len(results)} frames, calibrated {len(ok_res)} "
          f"({100 * len(ok_res) / max(1, len(results)):.0f}%)")
    if not ok_res:
        if worst_frame is not None:
            cv2.imwrite(os.path.join(args.out, "worst_frame.png"), worst_frame)
        print("No frame calibrated. Check --rotate, lighting, and that >=4 tile lines "
              "per direction are visible. Saved a failed frame to worst_frame.png")
        json.dump({"frames": results}, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
        return

    # --- stability: is the same real-world point mapped to the same pixel? ----
    # Re-run on the sampled frames is expensive, so reuse the best H as reference
    # and re-detect on a subset of frames, comparing projected grid points.
    _, best_cal, best_idx = best
    ref_grid = np.array([[x, y] for x in range(0, 181, 60) for y in range(0, 241, 60)], np.float32)
    ref_px = project_real_to_image(best_cal.H, ref_grid)

    shifts = [float(np.median(np.linalg.norm(project_real_to_image(H, ref_grid) - ref_px, axis=1)))
              for H in all_H]
    shifts = np.array(shifts)
    n_pts = [r["n_points"] for r in ok_res]
    rms = [r["rms_cm"] for r in ok_res]
    # shifts of ~60+ px-equivalents of a tile mean the origin/indexing jumped between frames
    print(f"points per frame: min {min(n_pts)}, median {int(np.median(n_pts))}, max {max(n_pts)}")
    print(f"reprojection error (cm): median {np.median(rms):.2f}, max {max(rms):.2f}")
    print(f"grid shift vs best frame (px): median {np.median(shifts):.1f}, 95th pct {np.percentile(shifts, 95):.1f}, max {shifts.max():.1f}")
    print("  (only meaningful if the camera is FIXED; for handheld/moving footage large values are expected)")

    cv2.imwrite(os.path.join(args.out, "best_frame.png"), best_cal.debug_frame)
    if worst_frame is not None:
        cv2.imwrite(os.path.join(args.out, "worst_frame.png"), worst_frame)
    json.dump({"pts_image": best_cal.pts_image.tolist(), "pts_real": best_cal.pts_real.tolist(),
               "H": best_cal.H.tolist(), "frame": best_idx},
              open(os.path.join(args.out, "chosen_calibration.json"), "w"), indent=2)
    json.dump({"frames": results,
               "stability": {"median_shift_px": float(np.median(shifts)),
                             "max_shift_px": float(shifts.max())}},
              open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(f"Saved results to {args.out}/ (best frame #{best_idx})")


if __name__ == "__main__":
    main()
