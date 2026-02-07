"""
MediaPipe Hand Gesture Test

Simple camera test with hand landmark visualization.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
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
        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')


        # Create hand landmarker options
        base_options = python.BaseOptions(model_asset_path = model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands = 2,
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence = 0.5,
            min_tracking_confidence = 0.5
        )
        
        # Create the hand landmarker
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

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

    def test_camera(self):
        """Test camera and gesture data collection."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")
        
        print("\nTesting Gesture Data Collection")
        print("Press 'Q'\n")
        
        previousTime = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Could not read frame.")
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
 
            
            # Convert to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Process with MediaPipe
            self.frame_timestamp_ms += 33  # Approximate 30fps
            results = self.detector.detect(mp_image)
            
            # Draw hand landmarks
            num_hands = 0
            if results.hand_landmarks:
                num_hands = len(results.hand_landmarks)
                for hand_landmarks in results.hand_landmarks:
                    self.draw_landmarks(frame, hand_landmarks)
            
            # Calculate FPS
            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime

            # Display status
            status = f"Hands detected: {num_hands}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0) if num_hands > 0 else (0, 0, 255), 2)
            
            cv2.putText(frame, str(int(fps))+" FPS", (5, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            
            cv2.putText(frame, "Press Q to exit", (10, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Hand Gesture Test', frame)
            
            # Handle keys
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Gesture Data Collection Test")
    
    tracker = GestureDataCollector()
    tracker.test_camera()

    