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

        if ts_ms <= last_ts:
            ts_ms = last_ts + 1
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

win_name = "Hand-Tracking (Verzerrungs-Effekt)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

HAND_COLORS = {"Left": (0, 165, 255), "Right": (255, 200, 0)}
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

start_time = time.time()
fps_timer  = time.time()
fps = 0
fps_count = 0

print("Hand-Tracking (Verzerrung) läuft... Drücke 'q' zum Beenden.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    fps_count += 1

    if time.time() - fps_timer >= 1.0:
        fps       = fps_count
        fps_count = 0
        fps_timer = time.time()

    ts_ms = int((time.time() - start_time) * 1000)

    try:
        frame_queue.put_nowait((img.copy(), ts_ms))
    except queue.Full:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put_nowait((img.copy(), ts_ms))

    with result_lock:
        result = latest_result

    # ── Effekt-Logik ─────────────────────────────────────────────────────────
    nh = 0
    all_hand_pts = []
    
    if result and result.hand_landmarks:
        nh = len(result.hand_landmarks)
        for i, lms in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].display_name if result.handedness else "?"
            color = HAND_COLORS.get(label, (200, 200, 200))
            pts   = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            all_hand_pts.append(pts)

            # Hand-Skelett zeichnen
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, pts[a], pts[b], color, 2)
            for pt in pts:
                cv2.circle(img, pt, 4, (255, 255, 255), cv2.FILLED)

        # Verzerrungseffekt zwischen den Händen
        if len(all_hand_pts) == 2:
            pts1 = np.array(all_hand_pts[0])
            pts2 = np.array(all_hand_pts[1])
            
            # Bereich zwischen den Händen definieren (Konvexe Hülle aller Punkte)
            combined_pts = np.vstack([pts1, pts2])
            hull = cv2.convexHull(combined_pts)
            
            # Maske für den Verzerrungsbereich
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            
            # Dynamische Verzerrung (Pixel-Shift)
            shift_x = int(5 * np.sin(time.time() * 5))
            shift_y = int(5 * np.cos(time.time() * 5))
            
            # Das Bild im Bereich der Maske leicht verschieben (Glass-Effekt)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            distorted = cv2.warpAffine(img, M, (w, h))
            
            # Nur im Maskenbereich das verschobene Bild einblenden
            mask_bool = mask > 0
            # Mix aus Original und verschobenem Bild für Transparenz
            img[mask_bool] = cv2.addWeighted(img[mask_bool], 0.7, distorted[mask_bool], 0.3, 0)
            
            # Die Fäden zusätzlich über die Verzerrung zeichnen
            tips = [4, 8, 12, 16, 20]
            for idx in tips:
                p1, p2 = pts1[idx], pts2[idx]
                cv2.line(img, p1, p2, (200, 255, 255), 1, cv2.LINE_AA)
                # "Energie-Pulse" auf den Fäden
                pulse_pos = (np.sin(time.time() * 10 + idx) + 1) / 2
                px = int(p1[0] * (1 - pulse_pos) + p2[0] * pulse_pos)
                py = int(p1[1] * (1 - pulse_pos) + p2[1] * pulse_pos)
                cv2.circle(img, (px, py), 3, (255, 255, 255), cv2.FILLED)

    # ── Status Banner ────────────────────────────────────────────────────────
    cv2.rectangle(img, (0, 0), (w, 45), (30, 30, 30), cv2.FILLED)
    cv2.putText(img, f"Haende: {nh}/2   FPS: {fps}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow(win_name, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

frame_queue.put((None, 0))
cap.release()
cv2.destroyAllWindows()
