import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# --- MediaPipe and Angle Calculation Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- Exercise Logic Processor ---

class ExerciseProcessor:
    """
    This class encapsulates all logic for a specific exercise.
    """
    def __init__(self, exercise_type):
        self.exercise_type = exercise_type
        self.counter = 0
        self.state = 'get_ready'
        self.feedback = 'GET READY'
        self.visibility_threshold = 0.8
        
        self.joint_time_series = {
            'main_angle': [],
            'form_angle': []
        }
        
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process_frame(self, frame):
        """Processes a single video frame for pose estimation and exercise logic."""
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        main_angle = 0
        form_angle = 0
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            if self.exercise_type == "Squats":
                main_angle, form_angle = self._process_squats(landmarks)
            elif self.exercise_type == "Push-ups":
                main_angle, form_angle = self._process_pushups(landmarks)
            # *** NEW: Added Bicep Curls ***
            elif self.exercise_type == "Bicep Curls":
                main_angle, form_angle = self._process_bicep_curls(landmarks)

            # --- Update Time Series ---
            self.joint_time_series['main_angle'].append(main_angle)
            self.joint_time_series['form_angle'].append(form_angle)
            self.joint_time_series['main_angle'] = self.joint_time_series['main_angle'][-100:]
            self.joint_time_series['form_angle'] = self.joint_time_series['form_angle'][-100:]

        except Exception as e:
            self.feedback = "NO BODY DETECTED"
            self.joint_time_series['main_angle'].append(None)
            self.joint_time_series['form_angle'].append(None)

        # --- Render UI Elements on Frame ---
        self._render_ui(image, main_angle, form_angle)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        return image

    def _process_squats(self, landmarks):
        # ... (This method is unchanged from Phase 2) ...
        # 1. Get coords
        l_shoulder_val = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_hip_val = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        l_knee_val = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        l_ankle_val = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_shoulder_val = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_hip_val = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_knee_val = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_ankle_val = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # 2. Check visibility
        is_visible = all(lm.visibility > self.visibility_threshold for lm in 
                         [l_hip_val, l_knee_val, l_ankle_val, r_hip_val, r_knee_val, r_ankle_val])

        if not is_visible:
            self.feedback = "BODY NOT FULLY VISIBLE"
            return 0, 0

        l_hip, l_knee, l_ankle = [l_hip_val.x, l_hip_val.y], [l_knee_val.x, l_knee_val.y], [l_ankle_val.x, l_ankle_val.y]
        r_hip, r_knee, r_ankle = [r_hip_val.x, r_hip_val.y], [r_knee_val.x, r_knee_val.y], [r_ankle_val.x, r_ankle_val.y]
        l_shoulder, r_shoulder = [l_shoulder_val.x, l_shoulder_val.y], [r_shoulder_val.x, r_shoulder_val.y]
        
        # 3. Calculate angles
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        avg_knee_angle = (l_knee_angle + r_knee_angle) / 2
        
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        avg_hip_angle = (l_hip_angle + r_hip_angle) / 2

        # 4. Form check
        self.feedback = "GOOD FORM"
        if avg_hip_angle < 75:
            self.feedback = "KEEP CHEST UP"

        # 5. State machine
        if self.state == 'get_ready':
            if avg_knee_angle > 160:
                self.state = 'up'
                self.feedback = "READY! SQUAT DOWN"
        elif self.state == 'up':
            if avg_knee_angle < 100:
                self.state = 'down'
                self.feedback = "GOOD! GO UP"
        elif self.state == 'down':
            if avg_knee_angle > 160:
                self.counter += 1
                self.state = 'up'
                self.feedback = "REP COUNTED!"

        return avg_knee_angle, avg_hip_angle

    def _process_pushups(self, landmarks):
        # ... (This method is unchanged from Phase 2) ...
        # 1. Get coords
        l_shoulder_val = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_elbow_val = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        l_wrist_val = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        l_hip_val = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        l_ankle_val = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_shoulder_val = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_elbow_val = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        r_wrist_val = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        r_hip_val = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_ankle_val = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # 2. Check visibility
        is_visible = all(lm.visibility > self.visibility_threshold for lm in 
                         [l_shoulder_val, l_elbow_val, l_hip_val, r_shoulder_val, r_elbow_val, r_hip_val])
        
        if not is_visible:
            self.feedback = "BODY NOT FULLY VISIBLE"
            return 0, 0

        l_shoulder, l_elbow, l_wrist = [l_shoulder_val.x, l_shoulder_val.y], [l_elbow_val.x, l_elbow_val.y], [l_wrist_val.x, l_wrist_val.y]
        l_hip, l_ankle = [l_hip_val.x, l_hip_val.y], [l_ankle_val.x, l_ankle_val.y]
        r_shoulder, r_elbow, r_wrist = [r_shoulder_val.x, r_shoulder_val.y], [r_elbow_val.x, r_elbow_val.y], [r_wrist_val.x, r_wrist_val.y]
        r_hip, r_ankle = [r_hip_val.x, r_hip_val.y], [r_ankle_val.x, r_ankle_val.y]

        # 3. Calculate angles
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2

        l_back_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
        r_back_angle = calculate_angle(r_shoulder, r_hip, r_ankle)
        avg_back_angle = (l_back_angle + r_back_angle) / 2

        # 4. Form check
        self.feedback = "GOOD FORM"
        if avg_back_angle < 145:
            self.feedback = "KEEP YOUR BACK STRAIGHT"
        elif calculate_angle(l_hip, l_shoulder, l_elbow) > 65 or calculate_angle(r_hip, r_shoulder, r_elbow) > 65:
            self.feedback = "TUCK ELBOWS"

        # 5. State machine
        if self.state == 'get_ready':
            if avg_back_angle > 145 and avg_elbow_angle > 155:
                self.state = 'up'
                self.feedback = "READY! GO DOWN"
        elif self.state == 'up':
            if avg_elbow_angle < 90:
                self.state = 'down'
                self.feedback = "GOOD! PUSH UP"
        elif self.state == 'down':
            if avg_elbow_angle > 155:
                self.counter += 1
                self.state = 'up'
                self.feedback = "REP COUNTED!"

        return avg_elbow_angle, avg_back_angle

    # *** NEW METHOD ***
    def _process_bicep_curls(self, landmarks):
        # 1. Get coords
        l_shoulder_val = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_elbow_val = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        l_wrist_val = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        l_hip_val = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_shoulder_val = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_elbow_val = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        r_wrist_val = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        r_hip_val = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # 2. Check visibility (focus on arms)
        is_visible = all(lm.visibility > self.visibility_threshold for lm in 
                         [l_shoulder_val, l_elbow_val, l_wrist_val, r_shoulder_val, r_elbow_val, r_wrist_val])
        
        if not is_visible:
            self.feedback = "ARMS NOT FULLY VISIBLE"
            return 0, 0

        l_shoulder, l_elbow, l_wrist = [l_shoulder_val.x, l_shoulder_val.y], [l_elbow_val.x, l_elbow_val.y], [l_wrist_val.x, l_wrist_val.y]
        l_hip = [l_hip_val.x, l_hip_val.y]
        r_shoulder, r_elbow, r_wrist = [r_shoulder_val.x, r_shoulder_val.y], [r_elbow_val.x, r_elbow_val.y], [r_wrist_val.x, r_wrist_val.y]
        r_hip = [r_hip_val.x, r_hip_val.y]

        # 3. Calculate angles
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2

        # Form angle: Check for "swinging" by measuring shoulder angle
        l_shoulder_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_shoulder_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        avg_shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2

        # 4. Form check
        self.feedback = "GOOD FORM"
        # Check if shoulder angle is too wide (swinging back) or too narrow (swinging forward)
        if avg_shoulder_angle > 45 or avg_shoulder_angle < 15:
            self.feedback = "KEEP SHOULDERS STILL"

        # 5. State machine
        if self.state == 'get_ready':
            if avg_elbow_angle > 160:
                self.state = 'down'
                self.feedback = "READY! CURL UP"
        elif self.state == 'down':
            if avg_elbow_angle < 50:
                self.state = 'up'
                self.feedback = "GOOD! GO DOWN"
        elif self.state == 'up':
            if avg_elbow_angle > 160:
                self.counter += 1
                self.state = 'down'
                self.feedback = "REP COUNTED!"

        return avg_elbow_angle, avg_shoulder_angle

    def _render_ui(self, image, main_angle, form_angle):
        """Draws the feedback and debug info on the image."""
        
        # ... (Feedback box color logic is unchanged) ...
        if "KEEP" in self.feedback or "TUCK" in self.feedback: color = (0, 0, 255) # Red
        elif "GOOD" in self.feedback: color = (0, 150, 0) # Green
        elif "COUNTED" in self.feedback: color = (200, 100, 0) # Blue
        else: color = (128, 0, 0) # Dark Blue

        # ... (Main Status Box logic is unchanged) ...
        cv2.rectangle(image, (0, 0), (250, 73), (50, 50, 50), -1)
        cv2.putText(image, 'REPS', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STATUS', (130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, self.state.upper(), (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # ... (Dynamic Feedback Box logic is unchanged) ...
        (text_width, _), _ = cv2.getTextSize(self.feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        text_x = 250 + (390 - text_width) // 2
        cv2.rectangle(image, (250, 0), (640, 73), color, -1)
        cv2.putText(image, self.feedback, (text_x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        # *** MODIFIED: Added logic for Bicep Curls ***
        if self.exercise_type == "Squats":
            angle_text, form_text = "KNEE", "HIP"
        elif self.exercise_type == "Push-ups":
            angle_text, form_text = "ELBOW", "BACK"
        elif self.exercise_type == "Bicep Curls":
            angle_text, form_text = "ELBOW", "SHOULDER"
        else:
            angle_text, form_text = "MAIN", "FORM"
        
        cv2.putText(image, f"{form_text}: {int(form_angle)}", (15, 640 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"{angle_text}: {int(main_angle)}", (15, 640 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def get_summary(self):
        """Returns a dictionary summary of the workout."""
        return {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Exercise": self.exercise_type,
            "Reps": self.counter
        }