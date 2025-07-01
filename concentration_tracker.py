import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

score_history = deque(maxlen=10)
distraction = 0

def eye_aspect_ratio(landmarks, eye_points, image_w, image_h, frame):
    points = []
    for idx in eye_points:
        lm = landmarks[idx]
        x, y = int(lm.x * image_w), int(lm.y * image_h)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        points.append((x, y))
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (A + B) / (2.0 * C)

def is_blinking(ear, threshold=0.2):
    return ear < threshold

def get_head_pose_score(landmarks, image_w, image_h):
    nose = landmarks[1]
    x, y = nose.x * image_w, nose.y * image_h
    dist = np.linalg.norm([x - image_w / 2, y - image_h / 2])
    return 1.0 if dist < 0.3 * image_w else 0.0

def get_gaze_score(landmarks):
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    avg_x = (left_iris.x + right_iris.x) / 2
    return 1.0 if 0.4 < avg_x < 0.6 else 0.0

def compute_concentration_score(gaze, head_pose, blink):
    return round((0.4 * gaze + 0.4 * head_pose + 0.2 * (0 if blink else 1)) * 100, 2)

def draw_concentration_bar(score, frame):
    bar_x, bar_y, bar_width, bar_height = 30, 60, 200, 25
    fill = int(score * bar_width / 100)
    color = (0, 255, 0) if score >= 50 else (0, 100, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_height), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
    cv2.putText(frame, f"{score}%", (bar_x + bar_width + 10, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    results = face_mesh.process(frame_rgb)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 150, 255), thickness=1)
            )

            # Compute EAR
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h, frame)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h, frame)
            avg_ear = (left_ear + right_ear) / 2
            blink = is_blinking(avg_ear)

            gaze_score = get_gaze_score(landmarks)
            head_score = get_head_pose_score(landmarks, w, h)
            concentration = compute_concentration_score(gaze_score, head_score, blink)

            score_history.append(concentration)
            smooth_score = int(np.mean(score_history))

            # Draw info
            draw_concentration_bar(smooth_score, frame)
            cv2.putText(frame, f"Concentration: {smooth_score}%", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            if blink:
                cv2.putText(frame, "BLINKING", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)

            if smooth_score < 50:
                distraction += 1
            else:
                distraction = 0  # Reset if focused

            # Show distraction count
            cv2.putText(frame, f"Distractions: {distraction}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,100,255), 2)

            # Show status
            status = "ACTIVE" if distraction == 0 else "DISTRACTED"
            color = (0,255,0) if status == "ACTIVE" else (0,100,255)
            cv2.putText(frame, status, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    else:
        # If no face
        cv2.putText(frame, "NO FACE", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

    # Display full-size
    resized_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Concentration Tracker", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
