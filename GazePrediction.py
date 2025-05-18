import cv2
import mediapipe as mp
import pandas as pd
import pyautogui
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from calibration_eye_gaze import run_calibration
from handTracking import HandGestureControl

# Run calibration and get data
# gaze_data, screen_data = run_calibration()

# Create DataFrame
# data = pd.DataFrame(gaze_data, columns=['rel_x', 'rel_y'])
# data['screen_x'] = [x for x, y in screen_data]
# data['screen_y'] = [y for x, y in screen_data]

data = pd.read_csv("/Users/sarthaktyagi/Desktop/projects /ratinaChess/chess_calibration_data.csv")
# Normalize gaze data
scaler = StandardScaler()
X = scaler.fit_transform(data[['rel_x', 'rel_y']])
y_x = data['screen_x']
y_y = data['screen_y']

# Train regressors
model_x = RandomForestRegressor().fit(X, y_x)
model_y = RandomForestRegressor().fit(X, y_y)

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Hand gesture toggle
hgc = HandGestureControl()

# Eye landmarks
left_eye = {
    "iris": 468,
    "left_corner": 33,
    "right_corner": 133,
    "top_lid": 159,
    "bottom_lid": 145
}
right_eye = {
    "iris": 473,
    "left_corner": 362,
    "right_corner": 263,
    "top_lid": 386,
    "bottom_lid": 374
}

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    prediction_status = hgc.detect_toggle_gesture(frame)

    if prediction_status:
        cv2.putText(frame, "Gaze Tracking: ON", (60, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            try:
                l = face.landmark

                # Left eye
                l_iris = l[left_eye["iris"]]
                l_left = l[left_eye["left_corner"]]
                l_right = l[left_eye["right_corner"]]
                l_top = l[left_eye["top_lid"]]
                l_bottom = l[left_eye["bottom_lid"]]

                # Right eye
                r_iris = l[right_eye["iris"]]
                r_left = l[right_eye["left_corner"]]
                r_right = l[right_eye["right_corner"]]
                r_top = l[right_eye["top_lid"]]
                r_bottom = l[right_eye["bottom_lid"]]

                # Eye dimensions
                l_eye_width = l_right.x - l_left.x
                r_eye_width = r_right.x - r_left.x
                l_eye_height = l_bottom.y - l_top.y
                r_eye_height = r_bottom.y - r_top.y

                if l_eye_width == 0 or r_eye_width == 0 or l_eye_height == 0 or r_eye_height == 0:
                    continue

                # Normalized gaze features
                l_rel_x = (l_iris.x - l_left.x) / l_eye_width
                r_rel_x = (r_iris.x - r_left.x) / r_eye_width
                avg_rel_x = (l_rel_x + r_rel_x) / 2

                l_rel_y = (l_iris.y - l_top.y) / l_eye_height
                r_rel_y = (r_iris.y - r_top.y) / r_eye_height
                avg_rel_y = (l_rel_y + r_rel_y) / 2

                X_test = scaler.transform([[avg_rel_x, avg_rel_y]])

                pred_x = model_x.predict(X_test)[0]
                pred_y = model_y.predict(X_test)[0]

                pyautogui.moveTo(pred_x, pred_y)
                time.sleep(0.01)

            except IndexError:
                pass
    else:
        cv2.putText(frame, "Gaze Tracking: OFF", (60, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, "Capturing feed... press 'q' to quit", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Gaze Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



def predict_gaze_once():
    ret, frame = cap.read()
    if not ret:
        return None, None, False

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    prediction_status = hgc.detect_toggle_gesture(frame)

    if results.multi_face_landmarks and prediction_status:
        face = results.multi_face_landmarks[0]
        try:
            l = face.landmark

            # Eye dimensions
            l_iris = l[left_eye["iris"]]
            l_left = l[left_eye["left_corner"]]
            l_right = l[left_eye["right_corner"]]
            l_top = l[left_eye["top_lid"]]
            l_bottom = l[left_eye["bottom_lid"]]

            r_iris = l[right_eye["iris"]]
            r_left = l[right_eye["left_corner"]]
            r_right = l[right_eye["right_corner"]]
            r_top = l[right_eye["top_lid"]]
            r_bottom = l[right_eye["bottom_lid"]]

            l_eye_width = l_right.x - l_left.x
            r_eye_width = r_right.x - r_left.x
            l_eye_height = l_bottom.y - l_top.y
            r_eye_height = r_bottom.y - r_top.y

            if l_eye_width == 0 or r_eye_width == 0 or l_eye_height == 0 or r_eye_height == 0:
                return None, None, False

            l_rel_x = (l_iris.x - l_left.x) / l_eye_width
            r_rel_x = (r_iris.x - r_left.x) / r_eye_width
            avg_rel_x = (l_rel_x + r_rel_x) / 2

            l_rel_y = (l_iris.y - l_top.y) / l_eye_height
            r_rel_y = (r_iris.y - r_top.y) / r_eye_height
            avg_rel_y = (l_rel_y + r_rel_y) / 2

            X_test = scaler.transform([[avg_rel_x, avg_rel_y]])

            pred_x = model_x.predict(X_test)[0]
            pred_y = model_y.predict(X_test)[0]

            return pred_x, pred_y, True

        except:
            return None, None, False

    return None, None, prediction_status
