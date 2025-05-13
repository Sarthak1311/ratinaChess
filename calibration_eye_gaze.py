import cv2 
import mediapipe as mp 
import time 
import pyautogui

# setup mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True )
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

# callibration point on the screen
screen_width , screen_height = pyautogui.size()

calibration_point = [
    (screen_width //4,screen_height //4), # top-left(25%,25%)
    (screen_width //2,screen_height //4), # top-center (50%,25%)
    (3*screen_width //4,screen_height //4), # top-right(75%,25%)
    (screen_width //4,screen_height //2), # mid-left(25%,50%)
    (screen_width //2,screen_height //2), # center(50%,50%)
    (3*screen_width //4,screen_height //2), # center-right(75%,50%)
    (screen_width //4,3* screen_height//4), #bottom-left (25%,75%)
    (screen_width //2,3* screen_height//4),#bottom-center(50%,75%)
    (3*screen_width //4,3* screen_height//4), # bottom-right(75%,75%)
]
# to store data
gaze_data=[]
screen_data=[]

cap=cv2.VideoCapture(0)

for (point_x ,point_y) in calibration_point:
    print(f"look at the points on the screen:{point_x},{point_y}")
    collected =0

    while collected<30:
        ret, frame = cap.read()
        print(frame.shape)
        if not ret:
            break

        frame = cv2.flip(frame,1)
        cvt_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = face_mesh.process(cvt_frame)
        fh,fw,_ = frame.shape

        # show dot 
        cv2.circle(frame,(point_x,point_y),10,(0,0,255),-1)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            try: 
                # get landmarks 
                l_iris = face.landmark[left_eye["iris"]]
                l_left= face.landmark[left_eye["left_corner"]]
                l_right = face.landmark[left_eye["right_corner"]]
                l_top = face.landmark[left_eye["top_lid"]]
                l_bottom = face.landmark[left_eye["bottom_lid"]]

                r_iris = face.landmark[right_eye["iris"]]
                r_left = face.landmark[right_eye["left_corner"]]
                r_right = face.landmark[right_eye["right_corner"]]
                r_top = face.landmark[right_eye["top_lid"]]
                r_bottom = face.landmark[right_eye["bottom_lid"]]

                # compute relative horizontal(x) position
                l_eye_width = l_right.x - l_left.x
                r_eye_width = r_right.x - r_left.x
                if l_eye_width == 0 or r_eye_width == 0:
                    continue

                l_rel_x = (l_iris.x - l_left.x) / l_eye_width
                r_rel_x = (r_iris.x - r_left.x) / r_eye_width
                avg_rel_x = (l_rel_x + r_rel_x) / 2

                # Compute relative vertical (y) position
                l_eye_height = l_bottom.y - l_top.y
                r_eye_height = r_bottom.y - r_top.y
                if l_eye_height == 0 or r_eye_height == 0:
                    continue

                l_rel_y = (l_iris.y - l_top.y) / l_eye_height
                r_rel_y = (r_iris.y - r_top.y) / r_eye_height
                avg_rel_y = (l_rel_y + r_rel_y) / 2
                # Store gaze features and screen point
                gaze_data.append([avg_rel_x, avg_rel_y])
                screen_data.append([point_x, point_y])
                collected += 1

            except IndexError:
                pass

        # Show frame
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Saving collected data 
import pandas as pd
df = pd.DataFrame(gaze_data, columns=['rel_x', 'rel_y'])
df['screen_x'] = [x for x, y in screen_data]
df['screen_y'] = [y for x, y in screen_data]
df.to_csv("calibration_data.csv", index=False)

print("Calibration complete. Data saved to 'calibration_data.csv'.")


