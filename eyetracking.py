import cv2
import mediapipe as mp

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,refine_landmarks=True)

# Setup camera capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (required by Mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get facial landmarks
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Drawing all the landmarks on the face (temporary debugging step)
            # mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            h, w, _ = frame.shape
            right_eye_landmarks = [7,33,246,161,160,159,158,157,173,133,155,154,163,144,145,153,468,469,470,471,472]
            for lm_idx in right_eye_landmarks:
                lm = face_landmarks.landmark[lm_idx]   
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                cv2.putText(frame, str(lm_idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


            left_eye_landmarks = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,473,474,475,476,477]
            for lm_idx in left_eye_landmarks:
                lm = face_landmarks.landmark[lm_idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                cv2.putText(frame, str(lm_idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
    else:
        print("No face detected!")

    # Display the frame with drawn eye landmarks
    cv2.imshow("Eye Landmarks", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
