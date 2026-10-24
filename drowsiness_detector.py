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

# Initialize Pygame for the continuous beep
pygame.mixer.init()
pygame.mixer.music.load('/System/Library/Sounds/Ping.aiff') 

# Updated Thresholds to reduce false alerts
EAR_THRESHOLD = 0.20       # Lowered from 0.25 (requires tighter eye closure)
EAR_CONSEC_FRAMES = 20     # Increased from 15 (waits slightly longer before alarming)
MAR_THRESHOLD = 0.6
MAR_CONSEC_FRAMES = 15    

# MediaPipe FaceMesh indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 13, 308, 14]

# Global state for audio threading
is_drowsy = False
alarm_on = False

def calculate_ear(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def calculate_mar(mouth_points):
    A = dist.euclidean(mouth_points[1], mouth_points[3])
    B = dist.euclidean(mouth_points[0], mouth_points[2])
    return A / (B + 1e-6)

def play_voice_alert():
    global is_drowsy
    phrases = [
        "Wake up! Don't sleep!",
        "There's a long journey ahead, wake up!",
        "Keep your eyes on the road!",
        "Danger! Please open your eyes!",
        "Wake up! Pull over if you feel sleepy!" # Fixed pronunciation
    ]
    # Voice keeps firing sentences as long as eyes remain closed
    while is_drowsy:
        os.system(f"say '{random.choice(phrases)}'")
        time.sleep(0.1)

def main():
    global is_drowsy, alarm_on
    
    # Set to 1 to bypass iPhone Continuity Camera. 
    # Change back to 0 if your iPhone is totally disconnected.
    cap = cv2.VideoCapture(1) 
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    blink_counter = 0
    frame_count = 0
    calibration_ear_list = []
    is_calibrated = False
    current_ear_thresh = EAR_THRESHOLD

    print("Starting camera... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Night Vision / Low Light Check
        raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_dark = np.mean(raw_gray) < 60
        
        # Apply brightening filters
        brightened = cv2.convertScaleAbs(frame, alpha=1.5, beta=40)
        gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed_frame = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        
        results = face_mesh.process(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        
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
                
                # Draw facial tracking points
                cv2.polylines(frame, [np.array(left_eye_pts)], True, (255, 100, 0), 1)
                cv2.polylines(frame, [np.array(right_eye_pts)], True, (255, 100, 0), 1)
                cv2.polylines(frame, [np.array(mouth_pts)], True, (255, 100, 0), 1)

                # Auto-Calibration Sequence (First 100 frames)
                if not is_calibrated:
                    frame_count += 1
                    calibration_ear_list.append(ear)
                    cv2.putText(frame, "CALIBRATING... LOOK FORWARD", (w//2 - 200, h//2), 
                                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
                    if frame_count >= 100:
                        avg_ear = sum(calibration_ear_list) / len(calibration_ear_list)
                        current_ear_thresh = avg_ear * 0.75 # Sets threshold to 75% of normal open-eye EAR
                        is_calibrated = True
                else:
                    # Logic for drowsy detection
                    if ear < current_ear_thresh:
                        blink_counter += 1
                        if blink_counter >= EAR_CONSEC_FRAMES:
                            current_drowsy_state = True
                            danger_reason = "EYES CLOSED"
                    else:
                        blink_counter = 0

                    if chin_y > 0.85: # Head dropping check
                        current_drowsy_state = True
                        danger_reason = "HEAD DROPPING"
                        
                    # Display metrics
                    cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Thresh: {current_ear_thresh:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update global state for the audio thread
        is_drowsy = current_drowsy_state

        # Handle UI and Alarms
        if is_drowsy:
            # Pulsing red screen overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1) 
            alpha = 0.4 + 0.2 * math.sin(time.time() * 12) 
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            cv2.putText(frame, f"DANGER: {danger_reason}!", (50, h//2), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 3)

            if not alarm_on:
                pygame.mixer.music.play(-1) # Start continuous beep
                threading.Thread(target=play_voice_alert, daemon=True).start() # Start voice
                alarm_on = True
        else:
            if alarm_on:
                pygame.mixer.music.stop() # Instantly cut off beep when eyes open
                alarm_on = False

        # Apply screen flashlight white border if room is dark
        if is_dark:
            frame = cv2.copyMakeBorder(frame, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cv2.putText(frame, "LOW LIGHT MODE ACTIVE", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Driver Drowsiness Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()

if __name__ == "__main__":
    main()