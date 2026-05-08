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

# ── Face Landmarker ──────────────────────────────────────────────────────────
face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=2,
    min_face_detection_confidence=0.4,
    min_face_presence_confidence=0.4,
    min_tracking_confidence=0.4,
    output_face_blendshapes=True   # Emotionen/Ausdrücke
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

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

# Gesichts-Kontur-Verbindungen (vereinfacht: Ovale um Augen, Mund, Gesichtsumriss)
# Gesichtsumriss Indices (aus MediaPipe Face Mesh)
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
LEFT_EYE  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]
RIGHT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
LIPS      = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61]

def draw_face_contour(img, pts, indices, color, thickness=1):
    for i in range(len(indices) - 1):
        a, b = indices[i], indices[i+1]
        if a < len(pts) and b < len(pts):
            cv2.line(img, pts[a], pts[b], color, thickness)

# Blendshapes die wir anzeigen wollen
EXPRESSIONS = ["mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight", "browInnerUp"]

# Stats
total = 0
start_time = time.time()
fps_timer  = time.time()
fps = 0
fps_count = 0

print("Hand + Gesichts-Tracking läuft... Drücke 'q' zum Beenden.")

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
    face_result = face_detector.detect_for_video(mp_img, ts_ms)

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

    # ── Gesicht zeichnen ─────────────────────────────────────────────────────
    for fi, face_lms in enumerate(face_result.face_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_lms]

        # Alle 478 Punkte tiny zeichnen
        for pt in pts:
            cv2.circle(img, pt, 1, (180, 255, 180), cv2.FILLED)

        # Konturen
        FACE_COLOR = (100, 255, 100)
        draw_face_contour(img, pts, FACE_OVAL, FACE_COLOR, 2)
        draw_face_contour(img, pts, LEFT_EYE,  (255, 255, 100), 1)
        draw_face_contour(img, pts, RIGHT_EYE, (255, 255, 100), 1)
        draw_face_contour(img, pts, LIPS,      (100, 150, 255), 2)

        # Blendshapes (Ausdrücke) anzeigen
        if face_result.face_blendshapes and fi < len(face_result.face_blendshapes):
            blendshapes = {b.category_name: b.score for b in face_result.face_blendshapes[fi]}
            y_off = 80
            for expr in EXPRESSIONS:
                val = blendshapes.get(expr, 0)
                bar = int(120 * val)
                label_short = expr.replace("mouth","").replace("eye","").replace("brow","")
                cv2.putText(img, f"{label_short[:12]}", (w - 200, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
                cv2.rectangle(img, (w - 80, y_off - 12), (w - 80 + bar, y_off - 2),
                              (100, 255, 150), cv2.FILLED)
                cv2.rectangle(img, (w - 80, y_off - 12), (w - 80 + 120, y_off - 2),
                              (80, 80, 80), 1)
                y_off += 22

    # ── Status Banner ────────────────────────────────────────────────────────
    nh = len(hand_result.hand_landmarks)
    nf = len(face_result.face_landmarks)
    cv2.rectangle(img, (0, 0), (w, 55), (30, 30, 30), cv2.FILLED)
    cv2.putText(img, f"Haende: {nh}/2   Gesichter: {nf}/2   FPS: {fps}",
                (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Hand + Gesichts-Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
