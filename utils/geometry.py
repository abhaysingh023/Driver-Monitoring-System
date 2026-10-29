import numpy as np

def get_distance(p1, p2):
    """Calculates Euclidean distance between two (x, y) points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_mar(landmarks, w, h):
    """
    Mouth Aspect Ratio (MAR) using MediaPipe Face Mesh:
    Top lip: 13, Bottom lip: 14, Left corner: 78, Right corner: 308
    """
    top = (int(landmarks[13].x * w), int(landmarks[13].y * h))
    bottom = (int(landmarks[14].x * w), int(landmarks[14].y * h))
    left = (int(landmarks[78].x * w), int(landmarks[78].y * h))
    right = (int(landmarks[308].x * w), int(landmarks[308].y * h))

    vertical_dist = get_distance(top, bottom)
    horizontal_dist = get_distance(left, right)

    if horizontal_dist == 0:
        return 0.0
    return vertical_dist / horizontal_dist

def extract_eye_box(frame, landmarks, eye_indices, w, h, padding=12):
    """
    Finds the bounding box around the eye landmarks and returns:
    1. cropped_eye: The image array ready for LightCNN
    2. (x1, y1, x2, y2): Coordinates to draw the box on screen
    """
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(w, max(xs) + padding)
    y2 = min(h, max(ys) + padding)

    cropped_eye = frame[y1:y2, x1:x2]
    return cropped_eye, (x1, y1, x2, y2)

def check_pose(pose_landmarks, w, h):
    """
    Uses MediaPipe Pose landmarks:
    Nose: 0, Left Shoulder: 11, Right Shoulder: 12
    """
    nose = (int(pose_landmarks[0].x * w), int(pose_landmarks[0].y * h))
    l_sh = (int(pose_landmarks[11].x * w), int(pose_landmarks[11].y * h))
    r_sh = (int(pose_landmarks[12].x * w), int(pose_landmarks[12].y * h))

    # 1. Shoulder Lean: vertical height difference between shoulders
    shoulder_tilt = abs(l_sh[1] - r_sh[1])

    # 2. Neck Slouch: vertical distance from nose to shoulder midpoint
    shoulder_mid_y = (l_sh[1] + r_sh[1]) // 2
    neck_height = abs(shoulder_mid_y - nose[1])

    return shoulder_tilt, neck_height, (nose, l_sh, r_sh)