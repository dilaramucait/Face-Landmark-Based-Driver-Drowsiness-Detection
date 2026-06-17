import cv2
import mediapipe as mp
import numpy as np
import pygame

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

# -----------------------------
# LANDMARKS
# -----------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 0, 17, 13, 14]

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# EAR FUNCTION
# -----------------------------
def eye_aspect_ratio(landmarks, eye_points, w, h):
    points = []

    for i in eye_points:
        lm = landmarks[i]
        points.append([
            int(lm.x * w),
            int(lm.y * h)
        ])

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

        points.append([
            int(lm.x * w),
            int(lm.y * h)
        ])

    A = np.linalg.norm(np.array(points[2]) - np.array(points[3]))
    B = np.linalg.norm(np.array(points[0]) - np.array(points[1]))

    return A / B

# -----------------------------
# FACE BOX
# -----------------------------
def draw_face_box(frame, landmarks, w, h, color):
    xs = []
    ys = []

    for lm in landmarks:
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    padding = 20

    cv2.rectangle(
        frame,
        (x_min - padding, y_min - padding),
        (x_max + padding, y_max + padding),
        color,
        2
    )

# -----------------------------
# LANDMARK DRAWING
# -----------------------------
def draw_landmarks(frame, landmarks, w, h):
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(
            frame,
            (x, y),
            1,
            (255, 255, 255),
            -1
        )

# -----------------------------
# STATUS PANEL
# -----------------------------
def draw_status_panel(frame, ear, mar, drowsy, yawning):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (15, 15),
        (260, 130),
        (30, 30, 30),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.6,
        frame,
        0.4,
        0,
        frame
    )

    if drowsy:
        status = "DROWSY"
        color = (0, 0, 255)

    elif yawning:
        status = "YAWNING"
        color = (0, 165, 255)

    else:
        status = "ACTIVE"
        color = (0, 255, 0)

    cv2.putText(
        frame,
        "DRIVER MONITOR",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    cv2.putText(
        frame,
        f"EAR: {ear:.2f}",
        (25, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"MAR: {mar:.2f}",
        (140, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

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

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    ear = 0
    mar = 0

    drowsy = False
    yawning = False

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        # EAR
        left_ear = eye_aspect_ratio(
            face_landmarks.landmark,
            LEFT_EYE,
            w,
            h
        )

        right_ear = eye_aspect_ratio(
            face_landmarks.landmark,
            RIGHT_EYE,
            w,
            h
        )

        ear = (left_ear + right_ear) / 2

        # MAR
        mar = mouth_aspect_ratio(
            face_landmarks.landmark,
            MOUTH,
            w,
            h
        )

        # Drowsiness logic
        if ear < 0.22:
            closed_frames += 1
        else:
            closed_frames = 0
            alarm_on = False

        # Yawning logic
        if mar > 0.60:
            yawn_frames += 1
        else:
            yawn_frames = 0

        drowsy = closed_frames > 60
        yawning = yawn_frames > 15

        # Box color
        if drowsy:
            box_color = (0, 0, 255)

        elif yawning:
            box_color = (0, 165, 255)

        else:
            box_color = (0, 255, 0)

        # Draw UI
        draw_face_box(
            frame,
            face_landmarks.landmark,
            w,
            h,
            box_color
        )

        draw_landmarks(
            frame,
            face_landmarks.landmark,
            w,
            h
        )

        draw_status_panel(
            frame,
            ear,
            mar,
            drowsy,
            yawning
        )

    # -----------------------------
    # ALARM
    # -----------------------------
    if alarm_cooldown > 0:
        alarm_cooldown -= 1

    if drowsy and not alarm_on:

        # 2-5 seconds eyes closed
        if closed_frames < 150:
            alarm_sound.set_volume(0.4)

        # 5-10 seconds eyes closed
        elif closed_frames < 300:
            alarm_sound.set_volume(0.7)

        # More than 10 seconds eyes closed
        else:
            alarm_sound.set_volume(1.0)

        alarm_sound.play(-1)  # loop continuously
        alarm_on = True

    if drowsy:

        # Start alarm if not already playing
        if not alarm_on:
            alarm_sound.play(-1)  # loop forever
            alarm_on = True

        # Increase volume as eyes remain closed longer

        if closed_frames < 150:  # ~2-5 sec
            alarm_sound.set_volume(0.3)

        elif closed_frames < 300:  # ~5-10 sec
            alarm_sound.set_volume(0.6)

        elif closed_frames < 450:  # ~10-15 sec
            alarm_sound.set_volume(0.8)

        else:  # >15 sec
            alarm_sound.set_volume(1.0)

    else:
        alarm_sound.stop()
        alarm_on = False

    # -----------------------------
    # WARNINGS
    # -----------------------------
    if closed_frames > 300:

        cv2.putText(
            frame,
            "WAKE UP NOW!",
            (25, h - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            4
        )

    elif closed_frames > 150:

        cv2.putText(
            frame,
            "SEVERE DROWSINESS!",
            (25, h - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 165, 255),
            3
        )

    elif drowsy:

        cv2.putText(
            frame,
            "DROWSINESS DETECTED!",
            (25, h - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

    elif yawning:

        cv2.putText(
            frame,
            "YAWNING DETECTED",
            (25, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            3
        )

    cv2.imshow(
        "Driver Drowsiness Detection System",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()