import cv2
import mediapipe as mp
import numpy as np
import winsound

# -----------------------------
# LANDMARKS
# -----------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 0, 17, 13, 14]

# -----------------------------
# MEDIA PIPE SETUP
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# Clean HUD-style drawing (only contours)
def draw_face_landmarks(frame, face_landmarks):
    # Soft HUD colors
    face_color = (120, 255, 180)   # soft mint
    eye_color = (255, 255, 255)    # white highlight
    mouth_color = (180, 200, 255)  # soft blue

    # Base mesh (very subtle)
    mp_draw.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_draw.DrawingSpec(
            color=(80, 80, 80), thickness=1, circle_radius=0
        )
    )

    # Eyes (highlight stronger)
    mp_draw.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_draw.DrawingSpec(
            color=eye_color, thickness=2, circle_radius=1
        )
    )

    mp_draw.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_draw.DrawingSpec(
            color=eye_color, thickness=2, circle_radius=1
        )
    )

    # Mouth (medium emphasis)
    mp_draw.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_LIPS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_draw.DrawingSpec(
            color=mouth_color, thickness=2, circle_radius=1
        )
    )

    return frame

# -----------------------------
# EAR FUNCTION
# -----------------------------
def eye_aspect_ratio(landmarks, eye_points, w, h):
    points = []

    for i in eye_points:
        lm = landmarks[i]
        x, y = int(lm.x * w), int(lm.y * h)
        points.append([x, y])

    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    return (A + B) / (2.0 * C)

# -----------------------------
# MAR FUNCTION
# -----------------------------
def mouth_aspect_ratio(landmarks, mouth_points, w, h):
    points = []

    for i in mouth_points:
        lm = landmarks[i]
        x, y = int(lm.x * w), int(lm.y * h)
        points.append([x, y])

    A = np.linalg.norm(np.array(points[2]) - np.array(points[3]))
    B = np.linalg.norm(np.array(points[0]) - np.array(points[1]))

    return A / B

# -----------------------------
# DASHBOARD UI
# -----------------------------
def draw_dashboard(frame, ear, mar, closed_frames, yawn_frames):
    h, w, _ = frame.shape

    text = (245, 245, 245)
    muted = (180, 180, 180)
    alert = (0, 0, 255)
    ok = (0, 255, 120)

    drowsy = closed_frames > 60

    status = "ACTIVE"
    color = ok

    if drowsy:
        status = "DROWSY"
        color = alert

    x, y = 25, 40

    cv2.putText(frame, "DRIVER MONITOR", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text, 2)

    cv2.putText(frame, f"STATUS: {status}", (x, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"EAR: {ear:.2f}", (x, y + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, muted, 2)

    cv2.putText(frame, f"MAR: {mar:.2f}", (x, y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, muted, 2)

    cv2.putText(frame, f"YAWN: {'YES' if yawn_frames > 15 else 'NO'}", (x, y + 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, muted, 2)

    # progress bar
    bar_x, bar_y = x, y + 135
    bar_w, bar_h = 220, 6

    fill = int(max(0, min(1, (0.3 - ear) / 0.15)) * bar_w)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill, bar_y + bar_h), color, -1)

    # gauge
    center = (w - 110, 110)
    radius = 50

    cv2.circle(frame, center, radius, (80, 80, 80), 2)

    angle = int(min(max((1 - ear) * 300, 0), 270))
    cv2.ellipse(frame, center, (radius, radius),
                0, 135, 135 + angle, color, 2)

    cv2.putText(frame, f"{int((1 - ear) * 100)}%", (center[0] - 25, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text, 2)

    return frame

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

closed_frames = 0
yawn_frames = 0

alarm_on = False
alarm_cooldown = 0

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    ear = 0
    mar = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # 🔥 FACE LANDMARK UI (NEW)
            frame = draw_face_landmarks(frame, face_landmarks)

            # EAR
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            # MAR
            mar = mouth_aspect_ratio(face_landmarks.landmark, MOUTH, w, h)

            # EAR logic
            if ear < 0.22:
                closed_frames += 1
            else:
                closed_frames = 0
                alarm_on = False

            # YAWN logic
            if mar > 0.6:
                yawn_frames += 1
            else:
                yawn_frames = 0

    # -----------------------------
    # DROWSY STATE
    # -----------------------------
    drowsy = closed_frames > 60

    # -----------------------------
    # SMART ALARM (NO SPAM)
    # -----------------------------
    if alarm_cooldown > 0:
        alarm_cooldown -= 1

    if drowsy and not alarm_on and alarm_cooldown == 0:
        # strong wake-up pattern
        for _ in range(3):
            winsound.Beep(2200, 120)
            winsound.Beep(2800, 120)

        alarm_on = True
        alarm_cooldown = 120

    if not drowsy:
        alarm_on = False
        alarm_cooldown = 0

    # -----------------------------
    # WARNINGS
    # -----------------------------
    if drowsy:
        cv2.putText(frame, "DROWSINESS DETECTED!", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if yawn_frames > 15:
        cv2.putText(frame, "YAWNING DETECTED", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # -----------------------------
    # UI
    # -----------------------------
    frame = draw_dashboard(frame, ear, mar, closed_frames, yawn_frames)

    cv2.imshow("Driver Drowsiness Detection System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()