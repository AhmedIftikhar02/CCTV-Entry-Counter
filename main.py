import os
import json
import time
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from age_classifier import AgeClassifier  # Imported our new modular age classifier

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_PATH = "videos/entrance03.avi"   # Path to the input CCTV video file
MODEL_PATH = "yolo11n.pt"
TRACKER_CONFIG = "custom_botsort.yaml"
CALIBRATION_DB = "calibrations.json"

CONFIDENCE = 0.22              # Lower threshold for early detection in shadowed areas
IMAGE_SIZE = 640               # Higher inference resolution for sharp detection of angled/distant boxes
MAX_FRAME_WIDTH = 960
TARGET_PROCESS_FPS = 8.0       # Target processing FPS for consistent real-time playback without frame drops
SAVE_DEBUG_VIDEO = False
OUTPUT_VIDEO_PATH = "output_debug.mp4"

# Default fallback line (normalized coordinates)
DEFAULT_LINE_START = (0.25, 0.72)
DEFAULT_LINE_END   = (0.75, 0.75)

# STRICT EXIT-BLOCKING & ENTRY VALIDATION THRESHOLDS
SEGMENT_MARGIN_T       = 0.10   # Allows slightly angled entries near left/right line endpoints
STRICT_OUTSIDE_PX      = -3.0   # Minimum signed distance (px) on the outside of the boundary
THRESHOLD_SPAWN_MAX_PX = 10.0   # Max distance (px) for tracks emerging directly at the threshold (e.g., behind pillars)
DEEP_INDOOR_ORIGIN_PX  = 12.0   # Tracks originating further inside (> +12px) are permanently ignored (blocks exits)
INSIDE_CONFIRM_PX      =  8.0   # Minimum signed distance (px) required on the inside of the boundary

MIN_PERP_TRAVEL_PX     = 14.0   # Minimum perpendicular crossing travel (px) for standard entries
LATE_SPAWN_TRAVEL_PX   = 20.0   # Stricter inward travel (px) required for tracks detected late at the threshold
MIN_FOOT_INWARD_PX     = 14.0   # Minimum foot displacement (px) along the inward entry vector
MIN_HEAD_INWARD_PX     = 12.0   # Minimum head displacement (px) along the inward entry vector (negative for exits)
MIN_HEAD_FOOT_RATIO    = 0.16   # Minimum head-to-foot inward movement ratio
MIN_BODY_ASPECT_RATIO  = 1.35   # Minimum height/width ratio to filter waist-clipped boxes during leg occlusions

# Tight Split-ID Duplicate Blocker
DUPLICATE_WINDOW_FRAMES = 3
DUPLICATE_RADIUS_PX     = 18.0

MIN_TRACK_AGE = 2
TRACK_TIMEOUT = 18

calibration_points = []


def mouse_callback(event, x, y, flags, param):
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 2:
            calibration_points.append((x, y))
            print(f"Point {len(calibration_points)} set: ({x}, {y})")


def compute_line_geometry(p1, p2, flip_dir=False):
    ax, ay = float(p1[0]), float(p1[1])
    bx, by = float(p2[0]), float(p2[1])
    if ax > bx:
        ax, ay, bx, by = bx, by, ax, ay

    dx = bx - ax
    dy = by - ay
    length = float(np.hypot(dx, dy))
    if length == 0:
        return 0.0, (0.0, 1.0)

    nx = -dy / length
    ny = dx / length
    if flip_dir:
        nx, ny = -nx, -ny
    return length, (nx, ny)


def draw_entry_line(img, p1, p2, normal, label="ENTRY LINE"):
    cv2.line(img, p1, p2, (0, 0, 255), 3)
    cv2.circle(img, p1, 5, (0, 255, 255), -1)
    cv2.circle(img, p2, 5, (0, 255, 255), -1)

    mid_x = int((p1[0] + p2[0]) / 2)
    mid_y = int((p1[1] + p2[1]) / 2)
    arrow_tip = (int(mid_x + normal[0] * 38), int(mid_y + normal[1] * 38))
    cv2.arrowedLine(img, (mid_x, mid_y), arrow_tip, (255, 255, 0), 2, tipLength=0.35)
    cv2.putText(img, "IN", (arrow_tip[0] + 5, arrow_tip[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    cv2.putText(img, label, (min(p1[0], p2[0]), max(min(p1[1], p2[1]) - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def analyze_point_to_segment(pt, p1, p2, line_length, normal):
    if line_length == 0:
        return 0.0, 0.5
    ax, ay = float(p1[0]), float(p1[1])
    bx, by = float(p2[0]), float(p2[1])
    if ax > bx:
        ax, ay, bx, by = bx, by, ax, ay

    px, py = pt
    dx = bx - ax
    dy = by - ay
    perp_dist = (px - ax) * normal[0] + (py - ay) * normal[1]
    seg_t = ((px - ax) * dx + (py - ay) * dy) / (line_length * line_length)
    return perp_dist, seg_t


def prepare_frame(frame, target_w, target_h):
    if frame.shape[1] != target_w or frame.shape[0] != target_h:
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return frame


def load_saved_calibration(video_key):
    if os.path.exists(CALIBRATION_DB):
        try:
            with open(CALIBRATION_DB, "r", encoding="utf-8") as f:
                data = json.load(f)
            if video_key in data:
                entry = data[video_key]
                return tuple(entry["p1"]), tuple(entry["p2"]), bool(entry.get("flip", False))
        except Exception:
            pass
    return None


def save_calibration(video_key, p1, p2, flip_dir):
    data = {}
    if os.path.exists(CALIBRATION_DB):
        try:
            with open(CALIBRATION_DB, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[video_key] = {"p1": list(p1), "p2": list(p2), "flip": bool(flip_dir)}
    with open(CALIBRATION_DB, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ==========================================
# INITIALIZATION
# ==========================================
model = YOLO(MODEL_PATH)
age_classifier = AgeClassifier()  # Initialize Age Classifier object
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Unable to open video source '{VIDEO_PATH}'.")
    exit()

orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
raw_fps = cap.get(cv2.CAP_PROP_FPS)
fps = raw_fps if (raw_fps and 1.0 <= raw_fps <= 120.0) else 25.0

if orig_width > MAX_FRAME_WIDTH:
    scale = MAX_FRAME_WIDTH / float(orig_width)
    width = MAX_FRAME_WIDTH
    height = int(round(orig_height * scale))
else:
    width = orig_width
    height = orig_height

frame_stride = max(1, int(round(fps / TARGET_PROCESS_FPS)))
effective_fps = fps / frame_stride

ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read video frame.")
    exit()

first_frame = prepare_frame(first_frame, width, height)
video_key = os.path.basename(VIDEO_PATH)
saved_calib = load_saved_calibration(video_key)

print("====================================")
print("  GENERIC CCTV ENTRY COUNTER (AGE)  ")
print("====================================")
print(f"Video      : {video_key} ({orig_width}x{orig_height} @ {fps:.1f} FPS)")
print(f"Processing : {width}x{height} | Stride 1/{frame_stride} (~{effective_fps:.1f} FPS)")
print("====================================")

setup_preview = first_frame.copy()
if saved_calib:
    sp1, sp2, sflip = saved_calib
    _, snorm = compute_line_geometry(sp1, sp2, sflip)
    draw_entry_line(setup_preview, sp1, sp2, snorm, "SAVED LINE")
else:
    dp1 = (int(width * DEFAULT_LINE_START[0]), int(height * DEFAULT_LINE_START[1]))
    dp2 = (int(width * DEFAULT_LINE_END[0]), int(height * DEFAULT_LINE_END[1]))
    _, dnorm = compute_line_geometry(dp1, dp2, False)
    draw_entry_line(setup_preview, dp1, dp2, dnorm, "DEFAULT LINE")

cv2.imshow("Setup - 'c' to Calibrate | SPACE to Start", setup_preview)
key = cv2.waitKey(0) & 0xFF

flip_dir = False
if key == ord("c"):
    calib_frame = first_frame.copy()
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    while True:
        temp_img = calib_frame.copy()
        for pt in calibration_points:
            cv2.circle(temp_img, pt, 5, (0, 255, 255), -1)

        if len(calibration_points) == 2:
            _, preview_norm = compute_line_geometry(calibration_points[0], calibration_points[1], flip_dir)
            draw_entry_line(temp_img, calibration_points[0], calibration_points[1], preview_norm, "CALIBRATED LINE")

        cv2.imshow("Calibration", temp_img)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("f") and len(calibration_points) == 2:
            flip_dir = not flip_dir
        elif k in (13, 32) and len(calibration_points) == 2:
            line_start = calibration_points[0]
            line_end = calibration_points[1]
            save_calibration(video_key, line_start, line_end, flip_dir)
            break

    cv2.destroyWindow("Calibration")
elif saved_calib:
    line_start, line_end, flip_dir = saved_calib
else:
    line_start = (int(width * DEFAULT_LINE_START[0]), int(height * DEFAULT_LINE_START[1]))
    line_end   = (int(width * DEFAULT_LINE_END[0]), int(height * DEFAULT_LINE_END[1]))
    flip_dir   = False

cv2.destroyAllWindows()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

line_length, entry_normal = compute_line_geometry(line_start, line_end, flip_dir)

out_writer = None
if SAVE_DEBUG_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, effective_fps, (width, height))

# ==========================================
# MAIN TRACKING & COUNTING LOOP
# ==========================================
tracks = {}
recent_counts = deque(maxlen=20)
raw_frame_idx = 0
frame_number = 0
total_entries = 0
young_entries = 0  # Category Counter for Young
old_entries = 0    # Category Counter for Old/Adult

print("\nENTRY COUNTER WITH AGE LOGS ACTIVE. Press 'q' to stop...")
playback_start_wall = time.perf_counter()

while True:
    if frame_stride > 1 and (raw_frame_idx % frame_stride != 0):
        if not cap.grab():
            break
        raw_frame_idx += 1
        continue

    ret, frame = cap.read()
    if not ret:
        break

    raw_frame_idx += 1
    frame_number += 1
    sec = raw_frame_idx / fps
    timestamp = f"{int(sec // 60):02d}:{sec % 60:05.2f}"

    frame = prepare_frame(frame, width, height)

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        classes=[0],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    draw_entry_line(frame, line_start, line_end, entry_normal, "ENTRY LINE")

    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0])

            box_w = max(x2 - x1, 1)
            box_h = max(y2 - y1, 1)
            aspect_ratio = box_h / box_w

            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            head_x = foot_x
            head_y = int(y1)
            center_x = foot_x
            center_y = int((y1 + y2) / 2)
            foot_pt = (foot_x, foot_y)

            perp_dist, seg_t = analyze_point_to_segment(
                foot_pt, line_start, line_end, line_length, entry_normal
            )
            within_door_span = (-SEGMENT_MARGIN_T <= seg_t <= 1.0 + SEGMENT_MARGIN_T)

            if track_id not in tracks:
                tracks[track_id] = {
                    "age": 0,
                    "last_seen": frame_number,
                    "max_box_h": box_h,
                    "first_center": (center_x, center_y),
                    "indoor_origin": (perp_dist > DEEP_INDOOR_ORIGIN_PX),
                    "saw_strict_outside": False,
                    "threshold_candidate": False,
                    "min_origin_perp": None,
                    "origin_foot": None,
                    "origin_head": None,
                    "is_moving_inward": False,
                    "counted_perp_dist": None,
                    "entry_counted": False,
                    "age_category": None
                }

            track = tracks[track_id]
            track["age"] += 1
            track["last_seen"] = frame_number
            if box_h > track["max_box_h"]:
                track["max_box_h"] = box_h

            at_frame_edge = (y2 >= height - 10) or (y1 <= 10) or (x1 <= 10) or (x2 >= width - 10)
            is_full_body = (
                (aspect_ratio >= MIN_BODY_ASPECT_RATIO and box_h >= 0.70 * track["max_box_h"])
                or at_frame_edge
            )

            if track["entry_counted"] and track["counted_perp_dist"] is not None:
                if (track["counted_perp_dist"] - perp_dist) > 25.0:
                    track["entry_counted"] = False
                    track["indoor_origin"] = True
                    track["saw_strict_outside"] = False
                    track["threshold_candidate"] = False

            if is_full_body and within_door_span and not track["indoor_origin"]:
                if perp_dist <= STRICT_OUTSIDE_PX:
                    track["saw_strict_outside"] = True
                    if track["min_origin_perp"] is None or perp_dist < track["min_origin_perp"]:
                        track["min_origin_perp"] = perp_dist
                    if track["origin_foot"] is None:
                        track["origin_foot"] = (foot_x, foot_y)
                        track["origin_head"] = (head_x, head_y)
                elif (
                    not track["saw_strict_outside"]
                    and not track["threshold_candidate"]
                    and track["age"] <= 2
                    and perp_dist <= THRESHOLD_SPAWN_MAX_PX
                ):
                    track["threshold_candidate"] = True
                    track["min_origin_perp"] = perp_dist
                    track["origin_foot"] = (foot_x, foot_y)
                    track["origin_head"] = (head_x, head_y)

                if track["origin_foot"] is not None and track["min_origin_perp"] is not None:
                    perp_travel = perp_dist - track["min_origin_perp"]

                    foot_dx = foot_x - track["origin_foot"][0]
                    foot_dy = foot_y - track["origin_foot"][1]
                    foot_inward = foot_dx * entry_normal[0] + foot_dy * entry_normal[1]

                    head_dx = head_x - track["origin_head"][0]
                    head_dy = head_y - track["origin_head"][1]
                    head_inward = head_dx * entry_normal[0] + head_dy * entry_normal[1]

                    net_dx = center_x - track["first_center"][0]
                    net_dy = center_y - track["first_center"][1]
                    overall_inward = net_dx * entry_normal[0] + net_dy * entry_normal[1]

                    track["is_moving_inward"] = (perp_travel >= 8.0 and overall_inward >= 8.0)

                    required_travel = (
                        MIN_PERP_TRAVEL_PX if track["saw_strict_outside"] else LATE_SPAWN_TRAVEL_PX
                    )
                    head_ratio = (head_inward / foot_inward) if foot_inward > 0 else 0.0

                    if (
                        not track["entry_counted"]
                        and (track["saw_strict_outside"] or track["threshold_candidate"])
                        and track["age"] >= MIN_TRACK_AGE
                        and perp_dist >= INSIDE_CONFIRM_PX
                        and perp_travel >= required_travel
                        and foot_inward >= MIN_FOOT_INWARD_PX
                        and head_inward >= MIN_HEAD_INWARD_PX
                        and overall_inward >= 12.0
                        and head_ratio >= MIN_HEAD_FOOT_RATIO
                    ):
                        is_duplicate = False
                        for prev_frame, prev_x, prev_y in recent_counts:
                            if (frame_number - prev_frame) <= DUPLICATE_WINDOW_FRAMES:
                                if np.hypot(foot_x - prev_x, foot_y - prev_y) < DUPLICATE_RADIUS_PX:
                                    is_duplicate = True
                                    break

                        track["entry_counted"] = True
                        track["counted_perp_dist"] = perp_dist
                        
                        # Classify Age upon successful entry counting
                        category = age_classifier.classify_age(frame, x1, y1, x2, y2)
                        track["age_category"] = category

                        if not is_duplicate:
                            total_entries += 1
                            if category == "Young":
                                young_entries += 1
                            else:
                                old_entries += 1

                            recent_counts.append((frame_number, foot_x, foot_y))
                            
                            # REAL-TIME LOG PRINT WITH AGE CATEGORY
                            print(
                                f"[{timestamp} | F:{raw_frame_idx:4d}] ENTRY: ID {track_id:3d} | "
                                f"Category: [{category.upper()}] | Total: {total_entries:2d} "
                                f"(Young: {young_entries}, Old: {old_entries})"
                            )

            if track["entry_counted"]:
                box_color = (0, 255, 0)
                cat_label = track.get("age_category", "Young")
                label = f"ID:{track_id} [{cat_label}]"
            elif track["is_moving_inward"] and not track["indoor_origin"]:
                box_color = (255, 200, 0)
                label = f"ID:{track_id} [ENTERING]"
            else:
                box_color = (160, 160, 160)
                label = f"ID:{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, foot_pt, 4, (0, 0, 255), -1)
            cv2.putText(frame, label, (x1, max(y1 - 6, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 2)

    inactive_ids = [tid for tid, trk in tracks.items() if (frame_number - trk["last_seen"]) > TRACK_TIMEOUT]
    for tid in inactive_ids:
        del tracks[tid]

    # Top Counter Banner (Showing Total, Young, and Old Counts Live)
    cv2.rectangle(frame, (15, 15), (460, 95), (0, 0, 0), -1)
    cv2.putText(frame, f"TOTAL: {total_entries}  ({timestamp})", (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Young: {young_entries} | Old/Adult: {old_entries}", (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if out_writer is not None:
        out_writer.write(frame)

    target_wall = playback_start_wall + (raw_frame_idx / fps)
    wait_ms = max(1, int((target_wall - time.perf_counter()) * 1000))

    cv2.imshow("CCTV Entry Counter", frame)
    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
        break

cap.release()
if out_writer is not None:
    out_writer.release()
cv2.destroyAllWindows()

print("\n====================================")
print("       FINAL EXECUTION REPORT       ")
print("====================================")
print(f"Video File            : {video_key}")
print(f"Final Total Entries   : {total_entries}")
print(f"  - Young Entries     : {young_entries}")
print(f"  - Old/Adult Entries : {old_entries}")
print("====================================\n")