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

# Schnips-Logik State
# Wir speichern pro Hand (0, 1), ob sie gerade "gespannt" ist
hand_primed = [False, False]
last_prime_time = [0, 0]
snap_display_until = 0

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

# ── Webcam ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

win_name = "Hand-Tracking (Snap Detection)"
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

print("Hand-Tracking (Snap Detection) läuft... Drücke 'q' zum Beenden.")

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

    # ── Effekt & Schnips Logik ───────────────────────────────────────────────
    all_hand_pts = []
    if result and result.hand_landmarks:
        for i, lms in enumerate(result.hand_landmarks):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            all_hand_pts.append(pts)
            
            # 1. Schnips-Erkennung (Daumen 4 zu Mittelfinger 12)
            thumb_tip = lms[4]
            middle_tip = lms[12]
            dist = np.sqrt((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)
            
            curr_time = time.time()
            # Phase 1: Drücken (nah beieinander)
            if dist < 0.04:
                hand_primed[i] = True
                last_prime_time[i] = curr_time
            # Phase 2: Schnelles Lösen (Abstand wächst schlagartig)
            elif hand_primed[i] and dist > 0.12:
                if curr_time - last_prime_time[i] < 0.5: # Schnipser muss schnell sein
                    snap_display_until = curr_time + 1.0 # 1 Sekunde einblenden
                hand_primed[i] = False
            
            # Falls zu viel Zeit vergeht, Prime-State löschen
            if hand_primed[i] and curr_time - last_prime_time[i] > 0.6:
                hand_primed[i] = False

            # Skelett zeichnen
            label = result.handedness[i][0].display_name if result.handedness else "?"
            color = HAND_COLORS.get(label, (200, 200, 200))
            for a, b in HAND_CONNECTIONS: cv2.line(img, pts[a], pts[b], color, 2)
            for pt in pts: cv2.circle(img, pt, 4, (255, 255, 255), cv2.FILLED)

        # Heat-Haze Segmente (wie zuvor)
        if len(all_hand_pts) == 2:
            pts1, pts2 = np.array(all_hand_pts[0]), np.array(all_hand_pts[1])
            tips_idx = [4, 8, 12, 16, 20]
            
            mask = np.zeros((h, w), dtype=np.uint8)
            for j in range(len(tips_idx) - 1):
                idx_a, idx_b = tips_idx[j], tips_idx[j+1]
                quad = np.array([pts1[idx_a], pts2[idx_a], pts2[idx_b], pts1[idx_b]], dtype=np.int32)
                cv2.fillPoly(mask, [quad], 255)
            
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            # Distort
            flex_x = np.zeros((h, w), dtype=np.float32)
            flex_y = np.zeros((h, w), dtype=np.float32)
            t_wave = time.time() * 20
            # Wir machen es nur für das ganze Bild (einfacher zu programmieren für das Update)
            # Aber wir blenden es nur per Maske ein
            # Performance-Hack: Wir nehmen ein statisches Grid
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            flex_x = xx + 10 * np.sin(2 * np.pi * yy / 40 + t_wave)
            flex_y = yy
            distorted = cv2.remap(img, flex_x, flex_y, cv2.INTER_LINEAR)
            distorted = cv2.GaussianBlur(distorted, (5, 5), 0)
            
            alpha = (mask.astype(float) / 255.0) * 0.8
            alpha = cv2.merge([alpha, alpha, alpha])
            blended = (1.0 - alpha) * img.astype(float) + alpha * distorted.astype(float)
            img = blended.astype(np.uint8)
            
            for idx in tips_idx:
                cv2.line(img, tuple(pts1[idx]), tuple(pts2[idx]), (240, 255, 255), 1, cv2.LINE_AA)

    # ── UI Einblendungen ─────────────────────────────────────────────────────
    # SNAP Anzeige (Oben Rechts)
    if time.time() < snap_display_until:
        txt = "SNAP!"
        txt_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_TRIPLEX, 2.5, 5)[0]
        # Position oben rechts mit etwas Padding
        pos_x = w - txt_size[0] - 40
        pos_y = txt_size[1] + 60
        # Schatten für bessere Sichtbarkeit
        cv2.putText(img, txt, (pos_x+4, pos_y+4), cv2.FONT_HERSHEY_TRIPLEX, 2.5, (0, 0, 0), 5)
        cv2.putText(img, txt, (pos_x, pos_y), cv2.FONT_HERSHEY_TRIPLEX, 2.5, (0, 200, 255), 5)

    # Status Banner
    cv2.rectangle(img, (0, 0), (w, 45), (30, 30, 30), cv2.FILLED)
    cv2.putText(img, f"Haende: {len(all_hand_pts)}/2   FPS: {fps}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow(win_name, img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

frame_queue.put((None, 0))
cap.release()
cv2.destroyAllWindows()
