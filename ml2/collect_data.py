"""
MediaPipe Hand Landmark Data Collection Script

Records hand gesture landmarks from webcam or video files and saves to CSV/NumPy format.
Each gesture is labeled and stored with normalized landmark coordinates.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from datetime import datetime


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
    
    def __init__(self, output_dir="data"):
        """Initialize MediaPipe and data collection settings."""
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        # Create hand landmarker
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage for collected data
        self.data = []
        self.labels = []
        self.frame_timestamp_ms = 0
        
    def extract_landmarks(self, hand_landmarks):
        """
        Extract and normalize hand landmarks.
        
        Args:
            hand_landmarks: List of NormalizedLandmark objects from MediaPipe 0.10+
        
        Returns:
            np.array: Flattened array of 21 landmarks (x, y, z) = 63 features
        """
        landmarks = []
        for landmark in hand_landmarks:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks and connections on the frame."""
        h, w, _ = frame.shape
        
        # Draw connections
        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (x, y), 7, (0, 255, 0), 2)
    
    def collect_from_webcam(self, gesture_name, num_samples=100):
        """
        Collect gesture samples from webcam in real-time.
        
        Args:
            gesture_name: Label for the gesture (e.g., "hello", "thank_you")
            num_samples: Number of samples to collect
        """
        cap = cv2.VideoCapture(0)
        collected = 0
        recording = False
        
        print(f"\n=== Collecting data for gesture: '{gesture_name}' ===")
        print(f"Target samples: {num_samples}")
        print("\nControls:")
        print("  SPACE - Start/Resume recording")
        print("  P - Pause recording")
        print("  Q - Quit and save")
        print("  K - Quit without saving\n")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                continue
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process frame with MediaPipe (increment timestamp for video mode)
            self.frame_timestamp_ms += 33  # ~30fps
            results = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
            
            # Draw hand landmarks
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    # Draw landmarks using custom drawing function
                    self.draw_landmarks(frame, hand_landmarks)
                    
                    # Collect data if recording
                    if recording and collected < num_samples:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        self.data.append(landmarks)
                        self.labels.append(gesture_name)
                        collected += 1
                        
                        if collected >= num_samples:
                            print(f"\nCollected {collected} samples!")
                            recording = False
            
            # Display status
            status = f"Recording: {recording} | Collected: {collected}/{num_samples}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0) if recording else (0, 0, 255), 2)
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Gesture Data Collection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to toggle recording
                recording = not recording
                print(f"Recording: {'ON' if recording else 'OFF'}")
            elif key == ord('p'):  # Pause
                recording = False
                print("Recording PAUSED")
            elif key == ord('q'):  # Quit and save
                break
            elif key == ord('k'):  # K to quit without saving
                print("Quitting without saving...")
                self.data = self.data[:-collected]
                self.labels = self.labels[:-collected]
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return collected
    
    def save_to_csv(self, filename=None):
        """Save collected data to CSV file."""
        if not self.data:
            print("No data to save!")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gesture_data_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create DataFrame
        data_array = np.array(self.data)
        columns = []
        for i in range(21):  # 21 landmarks
            columns.extend([f'x{i}', f'y{i}', f'z{i}'])
        
        df = pd.DataFrame(data_array, columns=columns)
        df['label'] = self.labels
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"\nData saved to: {filepath}")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Features per sample: {data_array.shape[1]}")
        
        # Print label distribution
        print("\nLabel distribution:")
        label_counts = pd.Series(self.labels).value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} samples")
        
        return filepath
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()


def main():
    """Example usage of the GestureDataCollector."""
    collector = GestureDataCollector(output_dir="data/gestures")
    
    # Define ASL gestures to collect
    gestures = [
        "hello",
        "thank_you",
        "please",
        "yes",
        "no",
        "help",
        "sorry",
        "welcome"
    ]
    
    print("ASL Gesture Data Collection")
    print("\nThis script will collect hand landmark data for ASL gestures.")
    print("You'll record multiple samples for each gesture.\n")
    
    # Interactive mode
    while True:
        print("\nOptions:")
        print("0. Test camera/hand detection (preview mode)")
        print("1. List predefined gestures")
        print("2. Collect data for a single gesture")
        print("3. Collect data for all predefined gestures")
        print("4. Save and exit")
        print("5. Exit without saving")
        
        choice = input("\nEnter choice (0-5): ").strip()
        
        if choice == "0":
            collector.test_camera()
            
        elif choice == "1":
            print("\nPredefined ASL Gestures:")
            for i, gesture in enumerate(gestures, 1):
                print(f"  {i}. {gesture}")
        
        elif choice == "2":
            print("\nPredefined gestures: " + ", ".join(gestures))
            gesture = input("Enter gesture name (or custom): ").strip().lower().replace(" ", "_")
            if not gesture:
                print("No gesture name provided. Skipping.")
                continue
            num_samples = input("Number of samples (default 100): ").strip()
            num_samples = int(num_samples) if num_samples else 100
            collector.collect_from_webcam(gesture, num_samples)
            
        elif choice == "3":
            for gesture in gestures:
                input(f"\nPress Enter to start collecting '{gesture}'...")
                collector.collect_from_webcam(gesture, num_samples=100)
                
        elif choice == "4":
            if collector.data:
                collector.save_to_csv()
                print("\nAll data saved successfully!")
            else:
                print("\nNo data collected. Nothing to save.")
            break
            
        elif choice == "5":
            print("Exiting without saving...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
