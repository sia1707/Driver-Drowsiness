import streamlit as st
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

# Page Config
st.set_page_config(page_title="Driver Drowsiness Detector", page_icon="🚗", layout="wide")

# Initialize Pygame Audio
pygame.mixer.init()
pygame.mixer.music.load('/System/Library/Sounds/Ping.aiff') 

# Thresholds
EAR_THRESHOLD, EAR_CONSEC_FRAMES = 0.25, 15    
MAR_THRESHOLD, MAR_CONSEC_FRAMES = 0.6, 15    

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

def play_voice_alert():
    phrases = [
        "Wake up! Don't sleep!",
        "There's a long journey ahead, wake up!",
        "Keep your eyes on the road!",
        "Danger! Please open your eyes!",
        "Wake up! Pull over if you are tired!"
    ]
    while st.session_state.get('is_drowsy', False):
        os.system(f"say '{random.choice(phrases)}'")
        time.sleep(0.1)

# UI Header
st.title("🚗 AI Driver Drowsiness & Safety Monitor")
st.markdown("Real-time telemetry, auto-calibration, AI voice warnings, and low-light enhancement.")

# Sidebar Controls
st.sidebar.header("System Controls")
run_app = st.sidebar.checkbox("Start Monitoring", value=False)
sensitivity = st.sidebar.slider("EAR Sensitivity Multiplier", 0.60, 0.90, 0.75, 0.05)

# Main layout columns
col1, col2 = st.metrics_col = st.columns([3, 1])
with col2:
    st.markdown("### Live Telemetry")
    ear_metric = st.empty()
    mar_metric = st.empty()
    status_metric = st.empty()

video_placeholder = col1.empty()

if run_app:
    cap = cv2.VideoCapture(1)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    blink_counter, yawn_counter = 0, 0
    frame_count = 0
    calibration_ear_list = []
    is_calibrated = False
    current_ear_thresh = EAR_THRESHOLD

    st.session_state['is_drowsy'] = False
    alarm_on = False

    while run_app:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera.")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Night Vision Check
        raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_dark = np.mean(raw_gray) < 60
        
        brightened = cv2.convertScaleAbs(frame, alpha=1.5, beta=40)
        gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        frame = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        current_drowsy_state = False
        danger_reason = ""
        ear, mar = 0.0, 0.0
        
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

                if not is_calibrated:
                    frame_count += 1
                    calibration_ear_list.append(ear)
                    cv2.putText(frame, "CALIBRATING... LOOK FORWARD", (w//2 - 180, h//2), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
                    if frame_count >= 100:
                        avg_ear = sum(calibration_ear_list) / len(calibration_ear_list)
                        current_ear_thresh = avg_ear * sensitivity
                        is_calibrated = True
                else:
                    if ear < current_ear_thresh:
                        blink_counter += 1
                        if blink_counter >= EAR_CONSEC_FRAMES:
                            current_drowsy_state = True
                            danger_reason = "EYES CLOSED"
                    else:
                        blink_counter = 0

                    if chin_y > 0.85:
                        current_drowsy_state = True
                        danger_reason = "HEAD DROPPING"

        st.session_state['is_drowsy'] = current_drowsy_state

        if st.session_state['is_drowsy']:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1) 
            alpha = 0.4 + 0.2 * math.sin(time.time() * 12) 
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            cv2.putText(frame, f"DANGER: {danger_reason}!", (50, h//2), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 3)

            if not alarm_on:
                pygame.mixer.music.play(-1) 
                threading.Thread(target=play_voice_alert, daemon=True).start()
                alarm_on = True
        else:
            if alarm_on:
                pygame.mixer.music.stop() 
                alarm_on = False

        if is_dark:
            frame = cv2.copyMakeBorder(frame, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # Update Sidebar/Metrics
        ear_metric.metric("EAR (Eye Ratio)", f"{ear:.2f}", f"Thresh: {current_ear_thresh:.2f}")
        mar_metric.metric("MAR (Mouth Ratio)", f"{mar:.2f}")
        status_metric.markdown(f"**Status:** {'🚨 DANGER' if st.session_state['is_drowsy'] else '🟢 NORMAL'}")

        # Convert OpenCV BGR to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    pygame.mixer.music.stop()