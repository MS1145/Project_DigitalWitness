import cv2
import mediapipe as mp
import csv
import numpy as np
import os

# --- CONFIGURATION (MATCHING YOUR SCREENSHOT) ---
# I used the 'r' before the string to handle Windows backslashes correctly
DATASET_ROOT = r"C:\Users\MSI\Documents\IIT\FY\FYP\Implementation\Project_DigitalWitness\DCSASS Dataset"

# The folders you have (Make sure you created 'Normal' and put videos in it!)
FOLDER_CRIME = "Shoplifting" 
FOLDER_NORMAL = "Normal"     
# -----------------------------------------------------------

OUTPUT_FILE = "training_data.csv"

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Setup CSV File
f = open(OUTPUT_FILE, 'w', newline='')
writer = csv.writer(f)
headers = ['label']
for val in range(1, 33+1): 
    headers += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
writer.writerow(headers)

def process_video(video_path, label_code):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return

    frame_count = 0
    # Print just the filename to keep it clean
    print(f"Processing: {os.path.basename(video_path)}")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # OPTIMIZATION: Process every 5th frame (Speed up by 5x)
        frame_count += 1
        if frame_count % 5 != 0: continue

        # Convert to RGB
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            if results.pose_landmarks:
                # Extract coordinates
                pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten())
                
                # Add Label (0 = Normal, 1 = Crime)
                pose_row.insert(0, label_code)
                writer.writerow(pose_row)
        except Exception as e:
            continue
            
    cap.release()

# --- MAIN EXECUTION ---
crime_path = os.path.join(DATASET_ROOT, FOLDER_CRIME)
normal_path = os.path.join(DATASET_ROOT, FOLDER_NORMAL)

# 1. Process Crime Videos
if os.path.exists(crime_path):
    print(f"Found Crime Folder: {crime_path}")
    files = [f for f in os.listdir(crime_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    if not files: print("WARNING: No video files found in Shoplifting folder!")
    
    for file in files:
        process_video(os.path.join(crime_path, file), 1)
else:
    print(f"ERROR: Could not find folder '{crime_path}'")

# 2. Process Normal Videos
if os.path.exists(normal_path):
    print(f"Found Normal Folder: {normal_path}")
    files = [f for f in os.listdir(normal_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    if not files: print("WARNING: No video files found in Normal folder! Did you download them?")
    
    for file in files:
        process_video(os.path.join(normal_path, file), 0)
else:
    print(f"ERROR: Could not find folder '{normal_path}'")

f.close()
print("------------------------------------------------")
print("DONE! Data saved to 'training_data.csv'")