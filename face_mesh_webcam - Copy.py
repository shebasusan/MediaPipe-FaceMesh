"""
Simple Face Mesh Project
-------------------------
Detects a face mesh (468 facial landmarks) in real time using your webcam.

Requirements:
    pip install opencv-python mediapipe

Run:
    python face_mesh_webcam.py

Press 'q' to quit.
"""

import cv2
import mediapipe as mp

# ---- Setup MediaPipe Face Mesh ----
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,          # detect up to 1 face (increase if needed)
    refine_landmarks=True,    # better detail around eyes/lips/iris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ---- Setup webcam ----
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting face mesh... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip for a mirror-like selfie view, convert BGR -> RGB for MediaPipe
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get face landmarks
    results = face_mesh.process(rgb_frame)

    # Draw the mesh if a face was found
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the tesselation (the mesh grid over the face)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style(),
            )
            # Draw the contours (eyes, eyebrows, lips, face outline)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style(),
            )

    cv2.imshow('Face Mesh - Press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
