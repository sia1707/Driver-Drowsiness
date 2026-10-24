import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import time
import math
import os
import random
import threading
import pygame

# ---------------------------------------------------------
# Setup & Configuration
# ---------------------------------------------------------
pygame.mixer.init()
pygame.mixer.music.load('/System/Library/Sounds/Ping.aiff') 

EAR_THRESHOLD = 0.25      
EAR_CONSEC_FRAMES = 15    
MAR_THRESHOLD = 0.6       
MAR_CONSEC_FRAMES = 15    

# Calibration Variables
frame_count = 0
calibration_ear_list = []
is_calibrated = False

# Counters, FPS, and Alarm State
blink_counter = 0
yawn_counter = 0
pTime = 0
is_drowsy = False  
alarm_on = False

# MediaPipe Indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 13, 308, 14]

def calculate_ear(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def calculate_mar(mouth_points):
    A = dist.euclidean(mouth_points[1], mouth_points[3])
    B = dist.euclidean(mouth_points[0], mouth_points[2])
    return A / (B + 1e-6)

# ---------------------------------------------------------
# AI Voice Thread
# ---------------------------------------------------------
def play_voice_alert():
    """Runs in the background and speaks phrases while drowsy"""
    global is_drowsy
    
    phrases = [
        "Wake up! Don't sleep!",
        "There's a long journey ahead, wake up!",
        "Keep your eyes on the road!",
        "Danger! Please open your eyes!",
        "Wake up! Pull over if you are tired!"
    ]
    
    while is_drowsy:
        phrase = random.choice(phrases)
        os.system(f"say '{phrase}'")
        time.sleep(0.1)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start Video Capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read from the camera.")
        break
        
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # ---------------------------------------------------------
    # 1. Analyze Room Brightness 
    # ---------------------------------------------------------
    # Check how dark the raw room is before we enhance it
    raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(raw_gray)
    is_dark = avg_brightness < 60  # If average pixel brightness is below 60, it's dark
    
    # ---------------------------------------------------------
    # 2. Software Night Vision Enhancer
    # ---------------------------------------------------------
    brightened = cv2.convertScaleAbs(frame, alpha=1.5, beta=40)
    gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    current_drowsy_state = False 
    danger_reason = "" 
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
            
            left_eye_pts = [landmarks[i] for i in LEFT_EYE]
            right_eye_pts = [landmarks[i] for i in RIGHT_EYE]
            mouth_pts = [landmarks[i] for i in MOUTH]
            
            ear = (calculate_ear(left_eye_pts) + calculate_ear(right_eye_pts)) / 2.0
            mar = calculate_mar(mouth_pts)
            chin_y = face_landmarks.landmark[152].y
            
            cv2.polylines(frame, [np.array(left_eye_pts)], True, (255, 100, 0), 1)
            cv2.polylines(frame, [np.array(right_eye_pts)], True, (255, 100, 0), 1)
            cv2.polylines(frame, [np.array(mouth_pts)], True, (255, 100, 0), 1)

            # Auto-Calibration Phase
            if not is_calibrated:
                frame_count += 1
                calibration_ear_list.append(ear)
                
                cv2.putText(frame, "SYSTEM CALIBRATING", (w//2 - 150, h//2 - 20), 
                            cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "PLEASE LOOK FORWARD", (w//2 - 160, h//2 + 20), 
                            cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
                
                if frame_count >= 100:
                    avg_ear = sum(calibration_ear_list) / len(calibration_ear_list)
                    EAR_THRESHOLD = avg_ear * 0.75
                    is_calibrated = True

            # Detection Phase
            else:
                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                    if blink_counter >= EAR_CONSEC_FRAMES:
                        current_drowsy_state = True 
                        danger_reason = "EYES CLOSED"
                else:
                    blink_counter = 0

                if chin_y > 0.85: 
                    current_drowsy_state = True  
                    danger_reason = "HEAD DROPPING"

                if mar > MAR_THRESHOLD:
                    yawn_counter += 1
                    if yawn_counter >= MAR_CONSEC_FRAMES:
                        cv2.putText(frame, "WARNING: YAWNING DETECTED", (20, h - 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    yawn_counter = 0
                    
            cv2.putText(frame, f"EAR: {ear:.2f} | THRESH: {EAR_THRESHOLD:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 50), 
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)

    is_drowsy = current_drowsy_state

    # ---------------------------------------------------------
    # 3. Cinematic UI Overlay & Audio Triggers
    # ---------------------------------------------------------
    if is_drowsy:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1) 
        alpha = 0.4 + 0.2 * math.sin(time.time() * 12) 
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        thickness = 15
        color = (0, 0, 255)
        cv2.line(frame, (0, 0), (100, 0), color, thickness)
        cv2.line(frame, (0, 0), (0, 100), color, thickness)
        cv2.line(frame, (w, 0), (w-100, 0), color, thickness)
        cv2.line(frame, (w, 0), (w, 100), color, thickness)
        cv2.line(frame, (0, h), (100, h), color, thickness)
        cv2.line(frame, (0, h), (0, h-100), color, thickness)
        cv2.line(frame, (w, h), (w-100, h), color, thickness)
        cv2.line(frame, (w, h), (w, h-100), color, thickness)

        font = cv2.FONT_HERSHEY_TRIPLEX
        alert_text = f"DANGER: {danger_reason}!"
        text_size = cv2.getTextSize(alert_text, font, 1.3, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        cv2.putText(frame, alert_text, (text_x, text_y), font, 1.3, (0, 0, 0), 8)
        cv2.putText(frame, alert_text, (text_x, text_y), font, 1.3, (255, 255, 255), 3)

        if not alarm_on:
            pygame.mixer.music.play(-1) 
            threading.Thread(target=play_voice_alert, daemon=True).start()
            alarm_on = True
    else:
        if alarm_on:
            pygame.mixer.music.stop() 
            alarm_on = False

    # FPS Counter
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (10, h - 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)

    # ---------------------------------------------------------
    # 4. The Screen Flashlight Hack
    # ---------------------------------------------------------
    # If the room is dark, pad the entire frame with a massive white border
    if is_dark:
        b_size = 250  # Thickness of the white light border in pixels
        frame = cv2.copyMakeBorder(frame, b_size, b_size, b_size, b_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # Add indicator text in the white border
        cv2.putText(frame, "AUTOMATIC SCREEN ILLUMINATION ON", (50, 100), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Emergency Driver Monitor", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
pygame.quit()