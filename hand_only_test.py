import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import threading
import queue
import math

# ── OneEuroFilter Klasse (Optimiert für weniger Lag) ──────────────────────────
class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.5, beta=0.05, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, dt):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0
            return x
        
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

class HandSmoother:
    def __init__(self):
        # Höhere Startwerte für min_cutoff und beta -> Weniger Geister-Effekt
        self.filters = [[OneEuroFilter(30, min_cutoff=1.5, beta=0.05) for _ in range(21*3)] for _ in range(2)]
        self.active = [False, False]

    def smooth(self, hand_idx, landmarks, dt):
        self.active[hand_idx] = True
        smoothed = []
        for i, lm in enumerate(landmarks):
            idx = i * 3
            sx = self.filters[hand_idx][idx](lm.x, dt)
            sy = self.filters[hand_idx][idx+1](lm.y, dt)
            sz = self.filters[hand_idx][idx+2](lm.z, dt)
            smoothed.append(type('Landmark', (), {'x': sx, 'y': sy, 'z': sz}))
        return smoothed

    def reset(self, hand_idx):
        for f in self.filters[hand_idx]:
            f.x_prev = None
        self.active[hand_idx] = False

# ── Shared State ─────────────────────────────────────────────────────────────
latest_result = None
result_lock   = threading.Lock()
frame_queue   = queue.Queue(maxsize=1)
smoother      = HandSmoother()

class SnapState:
    def __init__(self):
        self.is_primed = False
        self.prime_time = 0
        self.last_dist = 0

snap_states = [SnapState(), SnapState()]
last_snap_time = 0

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

win_name = "Hand-Tracking (Fast Smoothing)"
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
last_frame_time = time.time()
fps = 0
fps_count = 0

while True:
    curr_frame_t = time.time()
    dt = curr_frame_t - last_frame_time
    last_frame_time = curr_frame_t
    if dt <= 0: dt = 0.03

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

    # ── Effekt & Smooth Logik ────────────────────────────────────────────────
    all_hand_pts = []
    if result and result.hand_landmarks:
        num_found = len(result.hand_landmarks)
        for i in range(2):
            if i < num_found:
                raw_lms = result.hand_landmarks[i]
                lms = smoother.smooth(i, raw_lms, dt)
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                all_hand_pts.append(pts)
                
                hand_size = np.sqrt((lms[0].x - lms[9].x)**2 + (lms[0].y - lms[9].y)**2)
                dist = np.sqrt((lms[4].x - lms[12].x)**2 + (lms[4].y - lms[12].y)**2)
                state = snap_states[i]
                if dist < 0.18 * hand_size:
                    state.is_primed, state.prime_time = True, curr_frame_t
                elif state.is_primed:
                    dist_change = dist - state.last_dist
                    if dist > 0.6 * hand_size and dist_change > 0.1 * hand_size:
                        if curr_frame_t - state.prime_time < 0.4: last_snap_time = curr_frame_t
                        state.is_primed = False
                state.last_dist = dist
                if state.is_primed and curr_frame_t - state.prime_time > 0.5: state.is_primed = False

                color = HAND_COLORS.get(result.handedness[i][0].display_name if result.handedness else "?", (200, 200, 200))
                for a, b in HAND_CONNECTIONS: cv2.line(img, pts[a], pts[b], color, 2)
                for pt in pts: cv2.circle(img, pt, 4, (255, 255, 255), cv2.FILLED)
            else:
                smoother.reset(i)

        if len(all_hand_pts) == 2:
            pts1, pts2 = np.array(all_hand_pts[0]), np.array(all_hand_pts[1])
            tips_idx = [4, 8, 12, 16, 20]
            all_tips = []
            for j in range(len(tips_idx)-1):
                all_tips.extend([pts1[tips_idx[j]], pts2[tips_idx[j]], pts2[tips_idx[j+1]], pts1[tips_idx[j+1]]])
            x, y, bw, bh = cv2.boundingRect(np.array(all_tips, dtype=np.int32))
            x, y, bw, bh = max(0, x-15), max(0, y-15), min(w-x, bw+30), min(h-y, bh+30)
            if bw > 10 and bh > 10:
                roi = img[y:y+bh, x:x+bw].copy()
                distorted_roi = roi.copy()
                t_wave = time.time() * 25
                for r in range(bh):
                    offset = int(14 * np.sin(2 * np.pi * (r+y) / 25 + t_wave))
                    distorted_roi[r] = np.roll(roi[r], offset, axis=0)
                distorted_roi = cv2.GaussianBlur(distorted_roi, (7, 7), 0)
                mask_roi = np.zeros((bh, bw), dtype=np.uint8)
                for j in range(len(tips_idx) - 1):
                    idx_a, idx_b = tips_idx[j], tips_idx[j+1]
                    quad = np.array([pts1[idx_a], pts2[idx_a], pts2[idx_b], pts1[idx_b]], dtype=np.int32) - [x, y]
                    cv2.fillPoly(mask_roi, [quad], 255)
                mask_roi = cv2.GaussianBlur(mask_roi, (13, 13), 0)
                alpha = (mask_roi.astype(float) / 255.0) * 0.85
                alpha = cv2.merge([alpha, alpha, alpha])
                img[y:y+bh, x:x+bw] = ((1.0 - alpha) * img[y:y+bh, x:x+bw].astype(float) + alpha * distorted_roi.astype(float)).astype(np.uint8)
            for idx in tips_idx: cv2.line(img, tuple(pts1[idx]), tuple(pts2[idx]), (240, 255, 255), 1, cv2.LINE_AA)

    # UI
    cv2.rectangle(img, (0, 0), (w, 45), (30, 30, 30), cv2.FILLED)
    cv2.putText(img, f"Haende: {len(all_hand_pts)}/2   FPS: {fps}   REACTIVE SMOOTHING: ON", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow(win_name, img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

frame_queue.put((None, 0))
cap.release()
cv2.destroyAllWindows()
