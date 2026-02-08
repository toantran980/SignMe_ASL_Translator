"""
MediaPipe Hand Gesture Test

Simple camera test with hand landmark visualization.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker
import os
import numpy as np
import pandas as pd
import time

class GestureDataCollector:
    # Hand connections for drawing
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ] 
    
    def __init__(self):
        """Initialize MediaPipe hand detector."""
        # Get the model path
        #model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')

        # Create hand landmarker options
        hand_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=hand_options,
            num_hands = 2,
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence = 0.5,
            min_tracking_confidence = 0.5
        )

        # Create face landmarker options
        face_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5 
        )

        # Create the face landmarker
        self.face_detector = vision.FaceLandmarker.create_from_options(face_options)

        # Create the hand landmarker
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0
        
        # Initialize data storage
        self.data = []
        self.labels = []
        self.current_label = None

    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks and connections on the frame."""
        h, w, _ = frame.shape # height, width, channels
        
        # Draw connections
        for connection in self.HAND_CONNECTIONS:
            start, end= connection
            start = hand_landmarks[start]
            end = hand_landmarks[end]
            
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (x, y), 7, (0, 255, 0), 2)
    
    def draw_face_landmarks(self, frame, face_landmarks):
        """Draw face landmarks on the frame."""
        h, w, _ = frame.shape # height, width, channels
        
        for landmark in face_landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

    def process_landmarks(self, frame, mp_image):
        """Process both hand and face landmarks from a frame."""
        hand_results = self.detector.detect(mp_image)
        hand_data = None
        num_hands = 0
        
        if hand_results.hand_landmarks:
            num_hands = len(hand_results.hand_landmarks)
            for hand_landmarks in hand_results.hand_landmarks:
                self.draw_landmarks(frame, hand_landmarks)
            
            # Extract first hand data
            hand_data = [coord for landmark in hand_results.hand_landmarks[0] 
                        for coord in (landmark.x, landmark.y, landmark.z)]
        
        # Detect and process face
        face_results = self.face_detector.detect(mp_image)
        face_data = None
        
        if face_results.face_landmarks:
            for face_landmarks in face_results.face_landmarks:
                self.draw_face_landmarks(frame, face_landmarks)
            
            # Extract first face data
            face_data = [coord for landmark in face_results.face_landmarks[0] 
                        for coord in (landmark.x, landmark.y, landmark.z)]
        
        return hand_data, face_data, num_hands
    
    def save_gesture_data_csv(self, filename = None):
        """Save hand and face landmark data to CSV."""
        if not self.data:
            print("No data to save.")
            return
        
        # Create data/gestures directory if it doesn't exist
        os.makedirs('data/gestures', exist_ok=True)
        file_path = filename or f"data/gestures/gesture_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"

        # Create DataFrame
        data_array = np.array(self.data)
        columns = []

        # Hand landmarks (21)
        for i in range(21):
            columns.extend([f'hand_x{i}', f'hand_y{i}', f'hand_z{i}'])

        # Face landmarks (478)
        for i in range(478):
            columns.extend([f'face_x{i}', f'face_y{i}', f'face_z{i}'])
        
        df = pd.DataFrame(data_array, columns=columns)
        df['label'] = self.labels
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        print(f"\nData saved to: {file_path}")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Features per sample: {data_array.shape[1]}")
        
        # Print label distribution
        print("\nLabel distribution:")
        label_distribution = df['label'].value_counts()
        for label, count in label_distribution.items():
            print(f" {label}: {count} samples")
        return file_path

    def test_camera(self):
        """Test camera and gesture data collection."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")
        
        # Options for user to collect data
        print("\nTesting Data Collection List:")
        print("Press 'A'-'Z' or '0'-'9' to set label and start recording")
        print("Press 'SPACE' to capture frame with current label")
        print("Press 'C' to clear current label")
        print("Press 'Esc' to exit without saving")
        print("Press 'Q' to quit and save\n")
        
        previousTime = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Could not read frame.")
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
 
            # Convert to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Process with MediaPipe - combined hand and face detection
            self.frame_timestamp_ms += 33  # Approximate 30fps
            hand_data, face_data, num_hands = self.process_landmarks(frame, mp_image)

            # Calculate FPS
            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime

            # Display status
            hand_status = f"Hands detected: {num_hands}"
            cv2.putText(frame, hand_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0) if num_hands > 0 else (0, 0, 255), 2)
            
            face_status = "Face detected" if face_data else "No face"
            cv2.putText(frame, face_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 255) if face_data else (0, 0, 255), 2)
            
            cv2.putText(frame, str(int(fps))+" FPS", (520, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            
            # Display current label and sample count
            label_text = f"Label: {self.current_label if self.current_label else 'None'}"
            cv2.putText(frame, label_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
            
            samples_text = f"Samples: {len(self.data)}"
            cv2.putText(frame, samples_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, "Press Q to quit and save", (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Hand Gesture Test', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == 27:  # Esc key
                self.data = []
                self.labels = []
                print("Exit no saving.")
                break
            elif key == ord('c'):
                self.current_label = None
                print("Label cleared")
            elif (key >= ord('a') and key <= ord('z')) or (key >= ord('0') and key <= ord('9')):
                self.current_label = chr(key).upper()
                print(f"Label: {self.current_label}")
            elif key == ord(' '):
                if self.current_label and hand_data:
                    # Combine hand and face data
                    both_data = hand_data.copy()
                    if face_data:
                        both_data.extend(face_data)
                    else:
                        both_data.extend([0] * (478 * 3))  # 478 face landmarks * 3 coords
                    
                    self.data.append(both_data)
                    self.labels.append(self.current_label)
                    face_status = "face included" if face_data else "face excluded"
                    print(f"Captured sample {len(self.data)} for label '{self.current_label}' ({face_status})")
                elif not self.current_label:
                    print("Please set a label first (press A-Z or 0-9)")
                elif not hand_data:
                    print("No hand detected - please show your hand")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = GestureDataCollector()
    tracker.test_camera()
    tracker.save_gesture_data_csv()

    