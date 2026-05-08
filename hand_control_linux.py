import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import subprocess
from collections import deque

# ── Config ──────────────────────────────────────────────────────────────────
SMOOTH_WINDOW   = 7      # Wie viele Frames für den gleitenden Durchschnitt
DEADZONE_PCT    = 2      # Minimale Änderung (%) um Lautstärke zu updaten
CAM_INDEX       = 0
# ────────────────────────────────────────────────────────────────────────────

# 1. MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Audio (Linux amixer/pactl)
last_vol_set = -1

def set_volume(percentage):
    global last_vol_set
    percentage = int(max(0, min(100, percentage)))
    # Nur updaten wenn Änderung > DEADZONE_PCT
    if abs(percentage - last_vol_set) >= DEADZONE_PCT:
        # Versuche zuerst PulseAudio, dann ALSA
        ret = subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percentage}%"],
                             capture_output=True)
        if ret.returncode != 0:
            subprocess.run(["amixer", "-q", "sset", "Master", f"{percentage}%"], check=False)
        last_vol_set = percentage

# 3. Webcam Setup
cap = cv2.VideoCapture(CAM_INDEX)

# Gleitender Durchschnitt für Lautstärke
vol_history = deque(maxlen=SMOOTH_WINDOW)

print("Starte Hand-Tracking... Drücke 'q' zum Beenden.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)   # Kamera spiegeln – fühlt sich natürlicher an
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]

        # Alle Landmarks als Pixel-Koordinaten
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

        # Handgelenk (0) → Mittelfinger-Basis (9): Referenzgröße der Hand
        wrist      = np.array(pts[0])
        mid_base   = np.array(pts[9])
        hand_size  = np.linalg.norm(mid_base - wrist)   # px, abhängig vom Abstand zur Kamera

        # Daumenspitze (4) und Zeigefingerspitze (8)
        thumb      = np.array(pts[4])
        index      = np.array(pts[8])
        raw_dist   = np.linalg.norm(index - thumb)

        # Normalisierter Abstand (0 = ganz zu, 1 = weit auf)
        if hand_size > 0:
            norm_dist = raw_dist / hand_size
        else:
            norm_dist = 0

        # 0.1 = fast berühren, 0.9 = weit gespreizt → auf 0-100% mappen
        vol_raw = np.interp(norm_dist, [0.1, 0.9], [0, 100])

        # In den Gleitenden Durchschnitt eintragen
        vol_history.append(vol_raw)
        vol_smooth = np.mean(vol_history)

        set_volume(vol_smooth)

        # ── Visualisierung ──────────────────────────────────────────────────
        x1, y1 = pts[4]
        x2, y2 = pts[8]
        cv2.circle(img, (x1, y1), 14, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 14, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        vol_bar_y = int(np.interp(vol_smooth, [0, 100], [400, 150]))
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, vol_bar_y), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(vol_smooth)}%', (35, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Normalisierter Abstand anzeigen (Debug)
        cv2.putText(img, f'd={norm_dist:.2f}', (35, 490),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        # Kein Hand erkannt → kurze Info
        cv2.putText(img, "Keine Hand erkannt", (w//2 - 150, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 80, 255), 2)

    cv2.imshow("Hand Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
