import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np

# ── Hand Landmarker ──────────────────────────────────────────────────────────
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# ── Webcam ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

HAND_COLORS = {"Left": (0, 165, 255), "Right": (255, 200, 0)}
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Stats
total = 0
start_time = time.time()
fps_timer  = time.time()
fps = 0
fps_count = 0

print("Hand-Only Tracking läuft... Drücke 'q' zum Beenden.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    total    += 1
    fps_count += 1

    if time.time() - fps_timer >= 1.0:
        fps       = fps_count
        fps_count = 0
        fps_timer = time.time()

    ts_ms    = int((time.time() - start_time) * 1000)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    hand_result = hand_detector.detect_for_video(mp_img, ts_ms)

    # ── Hände zeichnen ───────────────────────────────────────────────────────
    for i, lms in enumerate(hand_result.hand_landmarks):
        label = hand_result.handedness[i][0].display_name if hand_result.handedness else "?"
        color = HAND_COLORS.get(label, (200, 200, 200))
        pts   = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], color, 2)
        for pt in pts:
            cv2.circle(img, pt, 5, (255,255,255), cv2.FILLED)
            cv2.circle(img, pt, 5, color, 1)
        cv2.putText(img, label, (pts[0][0]-20, pts[0][1]+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # ── Status Banner ────────────────────────────────────────────────────────
    nh = len(hand_result.hand_landmarks)
    cv2.rectangle(img, (0, 0), (w, 55), (30, 30, 30), cv2.FILLED)
    cv2.putText(img, f"Haende: {nh}/2   FPS: {fps}",
                (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Hand-Only Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
