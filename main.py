import cv2
import mediapipe as mp
import numpy as np

# --- Setup MediaPipe Tasks in IMAGE Mode (Guaranteed reliable, no timestamp sync drops) ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.IMAGE
)

def get_mar(face_lms, w, h):
    # Key inner lip landmarks: top (13), bottom (14), left (78), right (308)
    top = np.array([face_lms[13].x * w, face_lms[13].y * h])
    bot = np.array([face_lms[14].x * w, face_lms[14].y * h])
    left = np.array([face_lms[78].x * w, face_lms[78].y * h])
    right = np.array([face_lms[308].x * w, face_lms[308].y * h])

    vert = np.linalg.norm(top - bot)
    horiz = np.linalg.norm(left - right)
    return 0.0 if horiz == 0 else (vert / horiz)

def get_shoulder_metrics(pose_lms, w, h):
    # Left shoulder (11), Right shoulder (12)
    l_sh = np.array([pose_lms[11].x * w, pose_lms[11].y * h])
    r_sh = np.array([pose_lms[12].x * w, pose_lms[12].y * h])

    dx = r_sh[0] - l_sh[0]
    dy = r_sh[1] - l_sh[1]
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    if angle > 90:
        angle = abs(180 - angle)

    return angle, (int(l_sh[0]), int(l_sh[1])), (int(r_sh[0]), int(r_sh[1]))

# Open camera with DirectShow for Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# Yawn thresholds
MAR_NORMAL = 0.15
MAR_YAWN = 0.45
yawn_consecutive = 0
YAWN_FRAME_THRESH = 10  # ~0.3-0.5s sustained open mouth

print("Starting Yawn & Posture HUD... Press 'q' to quit.")

with FaceLandmarker.create_from_options(face_options) as face_detector, \
     PoseLandmarker.create_from_options(pose_options) as pose_detector:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert to RGB and wrap into MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect in IMAGE mode
        face_result = face_detector.detect(mp_image)
        pose_result = pose_detector.detect(mp_image)

        mar = 0.0
        yawn_prob = 0.0
        sh_angle = 0.0
        face_detected = False
        pose_detected = False
        is_yawning = False

        # 1. Process Face & Yawn
        if face_result.face_landmarks:
            face_detected = True
            face_lms = face_result.face_landmarks[0]
            mar = get_mar(face_lms, w, h)

            # Map MAR to 0.0 -> 1.0 probability
            scaled = (mar - MAR_NORMAL) / (MAR_YAWN - MAR_NORMAL)
            yawn_prob = float(np.clip(scaled, 0.0, 1.0))

            if mar > 0.38:
                yawn_consecutive += 1
            else:
                yawn_consecutive = max(0, yawn_consecutive - 1)

            if yawn_consecutive >= YAWN_FRAME_THRESH:
                is_yawning = True

            # Draw mouth points on your face
            for idx in [13, 14, 78, 308]:
                mx, my = int(face_lms[idx].x * w), int(face_lms[idx].y * h)
                cv2.circle(frame, (mx, my), 4, (0, 255, 255), -1)

        # 2. Process Shoulders & Posture
        if pose_result.pose_landmarks:
            pose_detected = True
            pose_lms = pose_result.pose_landmarks[0]
            sh_angle, pt_l, pt_r = get_shoulder_metrics(pose_lms, w, h)

            # Draw shoulder beam
            beam_color = (0, 255, 0) if sh_angle <= 10.0 else (0, 0, 255)
            cv2.line(frame, pt_l, pt_r, beam_color, 3)
            cv2.circle(frame, pt_l, 6, (255, 0, 0), -1)
            cv2.circle(frame, pt_r, 6, (255, 0, 0), -1)

        # ----------------- GUARANTEED TOP-LEFT HUD -----------------
        # Solid dark grey panel
        cv2.rectangle(frame, (10, 10), (390, 200), (25, 25, 25), -1)
        cv2.rectangle(frame, (10, 10), (390, 200), (0, 255, 255), 2)  # Bright yellow border

        # Line 1: Header / Sensor Status
        status_txt = f"Face: {'OK' if face_detected else 'LOST'} | Body: {'OK' if pose_detected else 'LOST'}"
        cv2.putText(frame, status_txt, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Line 2: Shoulder Tilt Angle
        angle_color = (0, 255, 0) if sh_angle <= 10.0 else (0, 0, 255)
        cv2.putText(frame, f"Shoulder Angle : {sh_angle:5.1f} deg", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, angle_color, 2)

        # Line 3: Raw Mouth Aspect Ratio (MAR)
        cv2.putText(frame, f"Mouth Ratio    : {mar:5.2f}", (20, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Line 4: Yawn Probability
        yawn_pct = int(yawn_prob * 100)
        prob_color = (0, 255, 0) if yawn_pct < 40 else (0, 165, 255) if yawn_pct < 75 else (0, 0, 255)
        cv2.putText(frame, f"Yawn Prob      : {yawn_pct:3d}%", (20, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, prob_color, 2)

        # Line 5: Progress Bar
        bar_x, bar_y, bar_w, bar_h = 20, 140, 250, 14
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
        fill_w = int(bar_w * yawn_prob)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), prob_color, -1)

        # Line 6: Alert Indicator
        if is_yawning:
            cv2.rectangle(frame, (20, 163), (270, 192), (0, 0, 255), -1)
            cv2.putText(frame, "! YAWN DETECTED !", (28, 184),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif sh_angle > 10.0:
            cv2.rectangle(frame, (20, 163), (280, 192), (0, 140, 255), -1)
            cv2.putText(frame, "! SHOULDER LEAN !", (28, 184),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "STATUS: DRIVER ALERT", (20, 184),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Driver Telemetry HUD", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()