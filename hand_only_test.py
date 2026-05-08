import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import threading
import queue

# ── Shared State ─────────────────────────────────────────────────────────────
latest_result = None
result_lock   = threading.Lock()
frame_queue   = queue.Queue(maxsize=1)

# Status Variablen
hand_primed = [False, False]
last_prime_time = [0, 0]
heat_haze_enabled = True

# Clap Detection State
last_palm_dist = 1.0
clap_explosion_until = 0
clap_explosion_pos = (0, 0)
clap_cooldown = 0

# ── MediaPipe Worker Thread ───────────────────────────────────────────────────
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

def inference_worker():
    global latest_result
    last_ts = 0
    while True:
        try:
            frame, ts_ms = frame_queue.get(timeout=1.0)
        except queue.Empty: continue
        if frame is None: break
        if ts_ms <= last_ts: ts_ms = last_ts + 1
        last_ts = ts_ms
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result  = hand_detector.detect_for_video(mp_img, ts_ms)
        with result_lock: latest_result = result

worker = threading.Thread(target=inference_worker, daemon=True)
worker.start()

# ── Webcam (320x240 für Speed) ───────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

win_name = "Hand-Tracking (Clap & Heat)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

HAND_COLORS = {"Left": (0, 165, 255), "Right": (255, 200, 0)}
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17),
]

start_time = time.time()
fps_timer  = time.time()
fps = 0
fps_count = 0

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    fps_count += 1
    if time.time() - fps_timer >= 1.0:
        fps, fps_count, fps_timer = fps_count, 0, time.time()

    ts_ms = int((time.time() - start_time) * 1000)
    try: frame_queue.put_nowait((img.copy(), ts_ms))
    except queue.Full:
        try: frame_queue.get_nowait()
        except queue.Empty: pass
        frame_queue.put_nowait((img.copy(), ts_ms))

    with result_lock: result = latest_result

    # ── Logik ────────────────────────────────────────────────────────────────
    all_hand_pts = []
    curr_t = time.time()
    
    if result and result.hand_landmarks:
        overlay = img.copy()
        
        # 1. Landmarks & Snap Toggle
        for i, lms in enumerate(result.hand_landmarks):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            all_hand_pts.append(pts)
            
            hand_size = np.sqrt((lms[0].x - lms[9].x)**2 + (lms[0].y - lms[9].y)**2)
            dist = np.sqrt((lms[4].x - lms[12].x)**2 + (lms[4].y - lms[12].y)**2)
            if dist < 0.18 * hand_size:
                hand_primed[i], last_prime_time[i] = True, curr_t
            elif hand_primed[i] and dist > 0.6 * hand_size:
                if curr_t - last_prime_time[i] < 0.4:
                    heat_haze_enabled = not heat_haze_enabled
                hand_primed[i] = False
            if hand_primed[i] and curr_t - last_prime_time[i] > 0.5: hand_primed[i] = False

            color = HAND_COLORS.get(result.handedness[i][0].display_name if result.handedness else "?", (200, 200, 200))
            for a, b in HAND_CONNECTIONS: cv2.line(overlay, pts[a], pts[b], color, 1)
            for pt in pts: cv2.circle(overlay, pt, 1, (255, 255, 255), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # 2. Clap Detection (Hände schlagen zusammen)
        if len(all_hand_pts) == 2:
            p1 = result.hand_landmarks[0][9] # Middle MCP als Zentrum
            p2 = result.hand_landmarks[1][9]
            dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            
            # Wenn Abstand schlagartig klein wird
            if dist < 0.12 and last_palm_dist > 0.25 and curr_t > clap_cooldown:
                clap_explosion_until = curr_t + 0.6
                # Mittelpunkt zwischen den Händen
                clap_explosion_pos = (int((p1.x + p2.x)/2 * w), int((p1.y + p2.y)/2 * h))
                clap_cooldown = curr_t + 1.0 # 1 Sekunde Cooldown
            
            last_palm_dist = dist

        # 3. Heat Haze
        if heat_haze_enabled and len(all_hand_pts) == 2:
            pts1, pts2 = np.array(all_hand_pts[0]), np.array(all_hand_pts[1])
            tips_idx = [4, 8, 12, 16, 20]
            all_tips = []
            for j in range(len(tips_idx)-1):
                all_tips.extend([pts1[tips_idx[j]], pts2[tips_idx[j]], pts2[tips_idx[j+1]], pts1[tips_idx[j+1]]])
            x, y, bw, bh = cv2.boundingRect(np.array(all_tips, dtype=np.int32))
            x, y, bw, bh = max(0, x-15), max(0, y-15), min(w-x, bw+30), min(h-y, bh+30)
            if bw > 5 and bh > 5:
                roi = img[y:y+bh, x:x+bw].copy()
                distorted_roi = roi.copy()
                t_wave = time.time() * 30
                for r in range(bh):
                    offset = int(16 * np.sin(2 * np.pi * (r+y) / 12 + t_wave))
                    distorted_roi[r] = np.roll(roi[r], offset, axis=0)
                distorted_roi = cv2.GaussianBlur(distorted_roi, (9, 9), 0)
                mask_roi = np.zeros((bh, bw), dtype=np.uint8)
                for j in range(len(tips_idx) - 1):
                    idx_a, idx_b = tips_idx[j], tips_idx[j+1]
                    quad = np.array([pts1[idx_a], pts2[idx_a], pts2[idx_b], pts1[idx_b]], dtype=np.int32) - [x, y]
                    cv2.fillPoly(mask_roi, [quad], 255)
                mask_roi = cv2.GaussianBlur(mask_roi, (11, 11), 0)
                alpha = (mask_roi.astype(float) / 255.0) * 0.9
                alpha = cv2.merge([alpha, alpha, alpha])
                img[y:y+bh, x:x+bw] = ((1.0 - alpha) * img[y:y+bh, x:x+bw].astype(float) + alpha * distorted_roi.astype(float)).astype(np.uint8)
            for idx in tips_idx: cv2.line(img, tuple(pts1[idx]), tuple(pts2[idx]), (240, 255, 255), 1, cv2.LINE_AA)

    # ── Explosion Effekt ──
    if curr_t < clap_explosion_until:
        remaining = clap_explosion_until - curr_t
        radius = int(120 * (1.0 - remaining/0.6)) # Expandiende Schockwelle
        alpha = remaining / 0.6 # Ausfaden
        
        ov = img.copy()
        cv2.circle(ov, clap_explosion_pos, radius, (255, 255, 255), 3) # Ring
        cv2.circle(ov, clap_explosion_pos, int(radius/2), (255, 255, 255), -1) # Kern
        cv2.addWeighted(ov, alpha, img, 1.0 - alpha, 0, img)

    # ── UI ──
    status_color = (0, 255, 0) if heat_haze_enabled else (0, 0, 255)
    cv2.rectangle(img, (0, 0), (w, 30), (30, 30, 30), cv2.FILLED)
    cv2.putText(img, f"FPS: {fps}   HEAT: {'ON' if heat_haze_enabled else 'OFF'}", (5, 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
    
    cv2.imshow(win_name, img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

frame_queue.put((None, 0))
cap.release()
cv2.destroyAllWindows()
