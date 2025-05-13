import mediapipe as mp 
import cv2

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

left_eye = {"iris": 468, "left_corner": 33, "right_corner": 133}
right_eye = {"iris": 473, "left_corner": 362, "right_corner": 263}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        try:
            # Left Eye
            left_iris = face_landmarks.landmark[left_eye["iris"]]
            left_corner = face_landmarks.landmark[left_eye["left_corner"]]
            right_corner = face_landmarks.landmark[left_eye["right_corner"]]

            # Right Eye
            right_iris = face_landmarks.landmark[right_eye["iris"]]
            right_corner_outer = face_landmarks.landmark[right_eye["right_corner"]]
            left_corner_inner = face_landmarks.landmark[right_eye["left_corner"]]

            # Normalize iris positions
            left_eye_width = right_corner.x - left_corner.x
            right_eye_width = right_corner_outer.x - left_corner_inner.x

            if left_eye_width == 0 or right_eye_width == 0:
                continue

            left_rel_x = (left_iris.x - left_corner.x) / left_eye_width
            right_rel_x = (right_iris.x - left_corner_inner.x) / right_eye_width

            avg_rel_x = (left_rel_x + right_rel_x) / 2

            # Draw iris positions
            cv2.circle(frame, (int(left_iris.x * w), int(left_iris.y * h)), 2, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_iris.x * w), int(right_iris.y * h)), 2, (0, 255, 0), -1)

            # Show position
            cv2.putText(frame, f"Avg Iris X: {avg_rel_x:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Adjusted thresholds
            if avg_rel_x < 0.48:
                direction = "Looking Left"
            elif avg_rel_x > 0.52:
                direction = "Looking Right"
            else:
                direction = "Looking Center"

            cv2.putText(frame, direction, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        except IndexError:
            print("Some landmarks not found!")

    else:
        cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Gaze Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
