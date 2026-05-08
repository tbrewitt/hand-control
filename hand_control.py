import cv2
import mediapipe as mp
import numpy as np
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# 1. MediaPipe Setup (Lokal, keine API-Keys!)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# 2. Audio Setup (Windows Lautstärke-Steuerung)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# 3. Webcam Setup
cap = cv2.VideoCapture(0)

print("Starte Hand-Tracking... Drücke 'q' zum Beenden.")

while True:
    success, img = cap.read()
    if not success:
        print("Kamera nicht gefunden.")
        break

    # Bild für MediaPipe vorbereiten (RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Punkte zeichnen
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Koordinaten für Daumen (ID 4) und Zeigefinger (ID 8) holen
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if len(lm_list) != 0:
                x1, y1 = lm_list[4][1], lm_list[4][2] # Daumenspitze
                x2, y2 = lm_list[8][1], lm_list[8][2] # Zeigefingerspitze

                # Kreis auf die Fingerspitzen
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Abstand berechnen
                length = np.hypot(x2 - x1, y2 - y1)
                
                # Abstand (ca. 20 - 200 Pixel) auf Lautstärke mappen
                vol = np.interp(length, [20, 200], [min_vol, max_vol])
                vol_bar = np.interp(length, [20, 200], [400, 150])
                vol_per = np.interp(length, [20, 200], [0, 100])

                volume.SetMasterVolumeLevel(vol, None)

                # Visualisierung der Lautstärke
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(vol_per)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    # Bild anzeigen
    cv2.imshow("Hand Control (Lokal)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
