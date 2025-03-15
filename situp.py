import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def main():
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose()
    
    counter = 0
    stage = None
    probability = 0.0
    
    cv2.namedWindow('Sit-Up Counter')  # Ensure window is created before setting mouse callback
    
    def click_event(event, x, y, flags, param):
        nonlocal counter
        if event == cv2.EVENT_LBUTTONDOWN:
            if 10 <= x <= 120 and 400 <= y <= 450:
                counter = 0
    
    cv2.setMouseCallback('Sit-Up Counter', click_event)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
            
            angle = calculate_angle(hip, knee, ankle)
            probability = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].visibility
            
            if angle > 160:
                stage = "up"
            if angle < 70 and stage == "up":
                stage = "down"
                counter += 1
                
            cv2.rectangle(image, (0, 0), (300, 80), (0, 0, 0), -1)
            cv2.putText(image, f'STAGE: {stage}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'REPS: {counter}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'PROB: {probability:.2f}', (150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
        cv2.rectangle(image, (10, 400), (120, 450), (0, 0, 255), -1)
        cv2.putText(image, "RESET", (20, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Sit-Up Counter', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
