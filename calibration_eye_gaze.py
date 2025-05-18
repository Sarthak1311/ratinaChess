import cv2
import mediapipe as mp
import time
import pandas as pd
import pygame
import numpy as np
import os

# Chessboard setup
WIDTH, HEIGHT = 800,800
SQUARE_SIZE = WIDTH // 8
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)

def draw_chessboard(screen):
    for row in range(8):
        for col in range(8):
            color = LIGHT if (row + col) % 2 == 0 else DARK
            rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)

def get_square_centers():
    centers = []
    for row in range(8):
        for col in range(8):
            x = col * SQUARE_SIZE + SQUARE_SIZE // 2
            y = row * SQUARE_SIZE + SQUARE_SIZE // 2
            centers.append((x, y))
    return centers

def run_calibration():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Calibration Board")

    # Set OpenCV window always on top and small
    # cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Webcam", 320, 240)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    # Landmarks for iris
    left_eye = {"iris": 468, "left_corner": 33, "right_corner": 133, "top_lid": 159, "bottom_lid": 145}
    right_eye = {"iris": 473, "left_corner": 362, "right_corner": 263, "top_lid": 386, "bottom_lid": 374}

    square_centers = get_square_centers()
    gaze_data, screen_data = [], []

    cap = cv2.VideoCapture(0)
 
    # Calibration loop
    for idx, (cx, cy) in enumerate(square_centers):
        collected = 0
        print(f"Looking at square {idx+1}/64: ({cx}, {cy})")

        dot_shown_time = time.time()

        while collected < 30:
            # ---- Pygame window with board + blue dot ----
            draw_chessboard(screen)
            pygame.draw.circle(screen, (0, 0, 255), (cx, cy), 10)
            pygame.display.flip()

            
            # ---- Webcam feed ----
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            # countdown
            for countdown in range(5,0,-1):
                print(countdown)
            if result.multi_face_landmarks:
                face = result.multi_face_landmarks[0]

                try:
                    # Left Eye
                    l = left_eye
                    l_iris = face.landmark[l["iris"]]
                    l_left = face.landmark[l["left_corner"]]
                    l_right = face.landmark[l["right_corner"]]
                    l_top = face.landmark[l["top_lid"]]
                    l_bottom = face.landmark[l["bottom_lid"]]

                    # Right Eye
                    r = right_eye
                    r_iris = face.landmark[r["iris"]]
                    r_left = face.landmark[r["left_corner"]]
                    r_right = face.landmark[r["right_corner"]]
                    r_top = face.landmark[r["top_lid"]]
                    r_bottom = face.landmark[r["bottom_lid"]]

                    # Horizontal ratio
                    l_w = l_right.x - l_left.x
                    r_w = r_right.x - r_left.x
                    if l_w == 0 or r_w == 0: continue

                    l_rx = (l_iris.x - l_left.x) / l_w
                    r_rx = (r_iris.x - r_left.x) / r_w
                    avg_rx = (l_rx + r_rx) / 2

                    # Vertical ratio
                    l_h = l_bottom.y - l_top.y
                    r_h = r_bottom.y - r_top.y
                    if l_h == 0 or r_h == 0: continue

                    l_ry = (l_iris.y - l_top.y) / l_h
                    r_ry = (r_iris.y - r_top.y) / r_h
                    avg_ry = (l_ry + r_ry) / 2

                    # Save data
                    gaze_data.append([avg_rx, avg_ry])
                    screen_data.append([cx, cy])
                    collected += 1
                    
                except IndexError:
                    continue

            cv2.putText(frame, f"Collecting: {collected}/30", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #cv2.imshow("Webcam", frame)

            # Early quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Allow pygame window to respond
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    
    # Save calibration data
    # df = pd.DataFrame(gaze_data, columns=["rel_x", "rel_y"])
    # df["screen_x"] = [x for x, y in screen_data]
    # df["screen_y"] = [y for x, y in screen_data]
    # df.to_csv("chess_calibration_data.csv", index=False)
    # print("✅ Calibration complete! Saved to 'chess_calibration_data.csv'.")
    return gaze_data ,screen_data

if __name__ == "__main__":
    run_calibration()
