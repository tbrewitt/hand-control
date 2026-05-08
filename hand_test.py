import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np

# ── Setup ────────────────────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,                           # ← Zwei Hände
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Farben pro Hand: Links = Orange, Rechts = Cyan
HAND_COLORS = {
    "Left":  (0, 165, 255),   # Orange
    "Right": (255, 200, 0),   # Cyan-Gelb
}

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Stats
total_frames    = 0
detected_frames = 0   # Frames wo mind. 1 Hand da war
start_time      = time.time()
fps_timer       = time.time()
fps             = 0
fps_counter     = 0

print("Zwei-Hand-Tracking läuft... Drücke 'q' zum Beenden.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    total_frames += 1
    fps_counter  += 1

    # FPS berechnen
    if time.time() - fps_timer >= 1.0:
        fps       = fps_counter
        fps_counter = 0
        fps_timer = time.time()

    timestamp_ms = int((time.time() - start_time) * 1000)
    img_rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result       = detector.detect_for_video(mp_image, timestamp_ms)

    num_hands = len(result.hand_landmarks)
    if num_hands > 0:
        detected_frames += 1

    # ── Jede erkannte Hand zeichnen ──────────────────────────────────────────
    for i, lms in enumerate(result.hand_landmarks):
        # Handedness (Links/Rechts) – nach flip ist es gespiegelt, daher tauschen
        if result.handedness and i < len(result.handedness):
            label = result.handedness[i][0].display_name  # "Left" oder "Right"
        else:
            label = "?"

        color = HAND_COLORS.get(label, (200, 200, 200))
        pts   = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

        # Verbindungen
        for a, b in CONNECTIONS:
            cv2.line(img, pts[a], pts[b], color, 2)

        # Landmarks
        for pt in pts:
            cv2.circle(img, pt, 6, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, pt, 6, color, 1)

        # Hand-Label beim Handgelenk
        cv2.putText(img, label, (pts[0][0] - 20, pts[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ── Status-Banner oben ───────────────────────────────────────────────────
    if num_hands == 2:
        banner_color = (0, 140, 0)
        status_text  = f"BEIDE HAENDE ERKANNT ({num_hands}/2)"
    elif num_hands == 1:
        banner_color = (0, 100, 200)
        status_text  = "1 HAND ERKANNT (1/2)"
    else:
        banner_color = (0, 0, 160)
        status_text  = "KEINE HAND ERKANNT"

    cv2.rectangle(img, (0, 0), (w, 55), banner_color, cv2.FILLED)
    cv2.putText(img, status_text, (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    # FPS
    cv2.putText(img, f"FPS: {fps}", (w - 120, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # ── Erkennungsrate + Balken unten ────────────────────────────────────────
    rate  = (detected_frames / total_frames * 100) if total_frames else 0
    bar_w = int(w * rate / 100)
    cv2.rectangle(img, (0, h - 30), (bar_w, h), (0, 180, 0), cv2.FILLED)
    cv2.rectangle(img, (0, h - 30), (w, h), (80, 80, 80), 2)
    cv2.putText(img,
                f"Hand-Erkennungsrate: {rate:.1f}%  ({detected_frames}/{total_frames} Frames)",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    # Legende
    cv2.circle(img, (w - 200, h - 65), 8, HAND_COLORS["Left"], cv2.FILLED)
    cv2.putText(img, "Links", (w - 185, h - 59),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, HAND_COLORS["Left"], 1)
    cv2.circle(img, (w - 120, h - 65), 8, HAND_COLORS["Right"], cv2.FILLED)
    cv2.putText(img, "Rechts", (w - 105, h - 59),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, HAND_COLORS["Right"], 1)

    cv2.imshow("Zwei-Hand-Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nErgebnis: {detected_frames}/{total_frames} Frames mit mind. 1 Hand erkannt ({rate:.1f}%)")
