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
        except queue.Empty:
            continue
        if frame is None:
            break
        if ts_ms <= last_ts: ts_ms = last_ts + 1
        last_ts = ts_ms
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result  = hand_detector.detect_for_video(mp_img, ts_ms)
        with result_lock:
            latest_result = result

worker = threading.Thread(target=inference_worker, daemon=True)
worker.start()

# ── Webcam ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

win_name = "Hand-Tracking (Extreme Heat Haze)"
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

print("Hand-Tracking (Heat Haze) läuft... Drücke 'q' zum Beenden.")

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    fps_count += 1
    if time.time() - fps_timer >= 1.0:
        fps = fps_count
        fps_count = 0
        fps_timer = time.time()

    ts_ms = int((time.time() - start_time) * 1000)
    try:
        frame_queue.put_nowait((img.copy(), ts_ms))
    except queue.Full:
        try: frame_queue.get_nowait()
        except queue.Empty: pass
        frame_queue.put_nowait((img.copy(), ts_ms))

    with result_lock: result = latest_result

    # ── Effekt-Logik ─────────────────────────────────────────────────────────
    all_hand_pts = []
    if result and result.hand_landmarks:
        for i, lms in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].display_name if result.handedness else "?"
            color = HAND_COLORS.get(label, (200, 200, 200))
            pts   = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            all_hand_pts.append(pts)
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, pts[a], pts[b], color, 2)
            for pt in pts: cv2.circle(img, pt, 4, (255, 255, 255), cv2.FILLED)

        if len(all_hand_pts) == 2:
            pts1, pts2 = np.array(all_hand_pts[0]), np.array(all_hand_pts[1])
            combined_pts = np.vstack([pts1, pts2])
            hull = cv2.convexHull(combined_pts)
            
            # 1. Bounding Box für Performance
            x, y, bw, bh = cv2.boundingRect(hull)
            # Clip to image boundaries
            x, y = max(0, x), max(0, y)
            bw, bh = min(w - x, bw), min(h - y, bh)
            
            if bw > 5 and bh > 5:
                roi = img[y:y+bh, x:x+bw].copy()
                
                # 2. Heat Haze Distortion (Wavy)
                # Wir verschieben jede Zeile basierend auf einer Sinuswelle
                distorted_roi = roi.copy()
                t = time.time() * 15
                for i in range(bh):
                    # Wellenbewegung: Amplitude 8px, Frequenz abhängig von Zeile und Zeit
                    offset = int(8 * np.sin(2 * np.pi * i / 30 + t))
                    distorted_roi[i] = np.roll(roi[i], offset, axis=0)
                
                # 3. Blur hinzufügen für den "heiße Luft" Look
                distorted_roi = cv2.GaussianBlur(distorted_roi, (7, 7), 0)
                
                # 4. Maske nur auf ROI anwenden
                roi_mask = np.zeros((bh, bw), dtype=np.uint8)
                hull_offset = hull - [x, y]
                cv2.fillConvexPoly(roi_mask, hull_offset, 255)
                
                # Weiche Kanten für die Maske
                roi_mask = cv2.GaussianBlur(roi_mask, (15, 15), 0)
                
                # Blending
                alpha = roi_mask.astype(float) / 255.0
                alpha = cv2.merge([alpha, alpha, alpha])
                
                img_roi = img[y:y+bh, x:x+bw].astype(float)
                dist_roi = distorted_roi.astype(float)
                
                # Blend: (1-alpha)*Original + alpha*Verzerrt
                # Wir erhöhen die Intensität des Effekts
                blended = (1.0 - alpha * 0.8) * img_roi + (alpha * 0.8) * dist_roi
                img[y:y+bh, x:x+bw] = blended.astype(np.uint8)
            
            # Fäden drüber zeichnen
            tips = [4, 8, 12, 16, 20]
            for idx in tips:
                p1, p2 = pts1[idx], pts2[idx]
                cv2.line(img, p1, p2, (220, 255, 255), 1, cv2.LINE_AA)

    # Status Banner
    cv2.rectangle(img, (0, 0), (w, 45), (30, 30, 30), cv2.FILLED)
    cv2.putText(img, f"Haende: {len(all_hand_pts)}/2   FPS: {fps}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow(win_name, img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

frame_queue.put((None, 0))
cap.release()
cv2.destroyAllWindows()
