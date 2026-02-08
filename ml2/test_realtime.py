"""
Real-time Gesture Recognition Testing

Test your trained model with live webcam feed.
Displays predictions with confidence scores in real-time.
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import sys

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class RealtimeGestureRecognizer:
    def __init__(self, model_path, metadata_path=None, model_type="sklearn"):
        """
        Initialize real-time gesture recognizer.
        
        Args:
            model_path: Path to trained model (.pkl, .h5, or .tflite)
            metadata_path: Path to metadata file (for sklearn models)
            model_type: "sklearn", "keras", or "tflite"
        """
        self.model_type = model_type
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Load model
        self.load_model(model_path, metadata_path)
        
    def load_model(self, model_path, metadata_path=None):
        """Load trained model and preprocessing objects."""
        print(f"Loading model: {model_path}")
        
        if self.model_type == "sklearn":
            # Load sklearn model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load metadata
            if metadata_path is None:
                metadata_path = model_path.replace('.pkl', '_metadata.pkl')
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.scaler = metadata['scaler']
            self.label_encoder = metadata['label_encoder']
            self.class_names = metadata['class_names']
            
        elif self.model_type == "keras":
            # Load Keras model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load metadata
            if metadata_path:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.scaler = metadata['scaler']
                self.class_names = metadata['class_names']
            else:
                print("Warning: No metadata provided. Using default preprocessing.")
                self.scaler = None
                self.class_names = None
                
        elif self.model_type == "tflite":
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Load metadata if provided
            if metadata_path:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.scaler = metadata.get('scaler')
                self.class_names = metadata.get('class_names')
            else:
                self.scaler = None
                self.class_names = None
        
        print(f"Model loaded successfully")
        if self.class_names:
            print(f"Classes: {', '.join(self.class_names)}")
    
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks from MediaPipe results."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def predict(self, landmarks):
        """
        Make prediction on hand landmarks.
        
        Returns:
            predicted_class, confidence, all_probabilities
        """
        # Reshape and preprocess
        landmarks = landmarks.reshape(1, -1)
        
        if self.scaler:
            landmarks = self.scaler.transform(landmarks)
        
        # Make prediction based on model type
        if self.model_type == "sklearn":
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(landmarks)[0]
                predicted_idx = np.argmax(probabilities)
                confidence = probabilities[predicted_idx]
            else:
                predicted_idx = self.model.predict(landmarks)[0]
                probabilities = None
                confidence = 1.0
            
            predicted_class = self.label_encoder.inverse_transform([predicted_idx])[0]
            
        elif self.model_type == "keras":
            probabilities = self.model.predict(landmarks, verbose=0)[0]
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            predicted_class = self.class_names[predicted_idx] if self.class_names else f"Class_{predicted_idx}"
            
        elif self.model_type == "tflite":
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                landmarks.astype(np.float32)
            )
            self.interpreter.invoke()
            probabilities = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            predicted_class = self.class_names[predicted_idx] if self.class_names else f"Class_{predicted_idx}"
        
        return predicted_class, confidence, probabilities
    
    def run(self, show_probabilities=True, confidence_threshold=0.6):
        """
        Run real-time gesture recognition.
        
        Args:
            show_probabilities: Display all class probabilities
            confidence_threshold: Minimum confidence to display prediction
        """
        cap = cv2.VideoCapture(0)
        
        print("\n" + "=" * 60)
        print("Real-time Gesture Recognition")
        print("=" * 60)
        print("\nControls:")
        print("  Q or ESC - Quit")
        print("  P - Toggle probability display")
        print("  SPACE - Take screenshot")
        print("\n")
        
        show_probs = show_probabilities
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                continue
            
            frame_count += 1
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks and predict gesture
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract landmarks and predict
                    landmarks = self.extract_landmarks(hand_landmarks)
                    predicted_class, confidence, probabilities = self.predict(landmarks)
                    
                    # Display prediction
                    if confidence >= confidence_threshold:
                        # Main prediction
                        text = f"{predicted_class.upper()}"
                        conf_text = f"{confidence*100:.1f}%"
                        
                        # Draw background rectangle
                        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
                        
                        # Draw text
                        cv2.putText(frame, text, (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        cv2.putText(frame, conf_text, (280, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        
                        # Draw probability bars
                        if show_probs and probabilities is not None:
                            y_offset = 100
                            for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities)):
                                # Draw bar
                                bar_width = int(prob * 300)
                                color = (0, 255, 0) if i == np.argmax(probabilities) else (100, 100, 100)
                                cv2.rectangle(frame, (10, y_offset), (10 + bar_width, y_offset + 20), color, -1)
                                
                                # Draw label
                                label_text = f"{class_name}: {prob*100:.1f}%"
                                cv2.putText(frame, label_text, (320, y_offset + 15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                y_offset += 30
                    else:
                        cv2.putText(frame, "Low Confidence", (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No hand detected", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Display FPS
            if frame_count % 30 == 0:
                fps_text = f"Frame: {frame_count}"
                cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Gesture Recognition', frame)
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('p'):  # Toggle probabilities
                show_probs = not show_probs
                print(f"Probability display: {'ON' if show_probs else 'OFF'}")
            elif key == ord(' '):  # Screenshot
                screenshot_path = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    import glob
    
    print("=" * 60)
    print("Real-time Gesture Recognition Testing")
    print("=" * 60)
    
    # Find available models
    models_dir = "models"
    
    if not os.path.exists(models_dir):

        print(f"\nError: Models directory not found: {models_dir}")
        print("Please train a model first using train_model.py")
        return
    
    pkl_models = glob.glob(os.path.join(models_dir, "*.pkl"))
    h5_models = glob.glob(os.path.join(models_dir, "*.h5"))
    tflite_models = glob.glob(os.path.join(models_dir, "*.tflite"))
    
    # Filter out metadata files
    pkl_models = [m for m in pkl_models if 'metadata' not in m]
    
    all_models = []
    if pkl_models:
        all_models.extend([("sklearn", m) for m in pkl_models])
    if h5_models:
        all_models.extend([("keras", m) for m in h5_models])
    if tflite_models:
        all_models.extend([("tflite", m) for m in tflite_models])
    
    if not all_models:
        print("\nNo trained models found!")
        print("Please train a model first using train_model.py")
        return
    
    # Display available models
    print("\nAvailable models:")
    for i, (model_type, path) in enumerate(all_models):
        print(f"  {i+1}. [{model_type}] {os.path.basename(path)}")
    
    # Select model
    choice = input(f"\nSelect model (1-{len(all_models)}): ").strip()
    try:
        idx = int(choice) - 1
        model_type, model_path = all_models[idx]
    except (ValueError, IndexError):
        print("Invalid choice!")
        return
    
    # Find metadata file
    metadata_path = None
    if model_type in ["sklearn", "keras"]:
        base_name = os.path.splitext(model_path)[0]
        potential_metadata = f"{base_name}_metadata.pkl"
        if os.path.exists(potential_metadata):
            metadata_path = potential_metadata
    
    # Initialize and run recognizer
    print(f"\nLoading model: {model_path}")
    recognizer = RealtimeGestureRecognizer(model_path, metadata_path, model_type)
    recognizer.run(show_probabilities=True, confidence_threshold=0.6)


if __name__ == "__main__":
    main()
