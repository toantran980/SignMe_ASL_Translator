"""
Gesture Recognition Model Training Script

Trains a classifier on MediaPipe hand landmarks using Scikit-learn or TensorFlow.
Supports multiple algorithms and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split  # , cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow imports (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available")


class GestureClassifier:
    def __init__(self, model_type="neural_network"):
        """
        Initialize gesture classifier.
        
        Args:
            model_type: Type of model to use
                # - "random_forest": Random Forest (default)
                # - "svm": Support Vector Machine
                # - "knn": K-Nearest Neighbors
                # - "gradient_boosting": Gradient Boosting
                - "neural_network": TensorFlow Neural Network (default)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
        
    def load_data(self, filepath, file_format="csv"):
        """
        Load training data from CSV or NumPy file.
        
        Args:
            filepath: Path to data file
            file_format: "csv" or "numpy"
            
        Returns:
            X, y: Features and labels
        """
        print(f"\nLoading data from: {filepath}")
        
        if file_format == "csv":
            df = pd.read_csv(filepath)
            X = df.drop('label', axis=1).values
            y = df['label'].values
            self.feature_names = df.drop('label', axis=1).columns.tolist()
        elif file_format == "numpy":
            data = np.load(filepath)
            X = data['X']
            y = data['y']
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Found {len(np.unique(y))} unique classes")
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2, random_state=42):
        """
        Preprocess and split data into train/test sets.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\nPreprocessing data...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Classes: {', '.join(self.class_names)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_model(self):
        """Create the specified model."""
        print(f"\nCreating {self.model_type} model...")
    
        if self.model_type == "neural_network":
            if not TENSORFLOW_AVAILABLE:
                raise ValueError("TensorFlow not available for neural network")
            self.model = self._create_neural_network()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def _create_neural_network(self, input_dim=63, num_classes=None):
        """Create a TensorFlow neural network model."""
        if num_classes is None:
            num_classes = len(self.class_names) if self.class_names is not None else 8
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for neural network)
            y_val: Validation labels (optional, for neural network)
        """
        print(f"\nTraining {self.model_type} model...")
        
        if self.model is None:
            self.create_model()
        
        if self.model_type == "neural_network":
            # Train neural network with validation
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=50,
                batch_size=32,
                verbose=1,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss' if X_val is not None else 'loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            return history

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        if self.model_type == "neural_network":
            y_pred = np.argmax(y_pred, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, self.class_names)
        
        return accuracy, y_pred
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'models/confusion_matrix_{timestamp}.png')
        print(f"\nConfusion matrix saved to models/confusion_matrix_{timestamp}.png")
    
    
    def save_model(self, filepath=None):
        """Save trained model and preprocessing objects."""
        os.makedirs('models', exist_ok=True)
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{self.model_type}_{timestamp}"
        else:
            base_name = filepath
        
        if self.model_type == "neural_network":
            # Save TensorFlow model
            model_path = f"models/{base_name}.h5"
            self.model.save(model_path)
            print(f"\nNeural network saved to: {model_path}")
        
        # Save preprocessing objects
        metadata = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'model_type': self.model_type
        }
        metadata_path = f"models/{base_name}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to: {metadata_path}")
        
        return model_path
    
    def load_model(self, model_path, metadata_path):
        """Load trained model and preprocessing objects."""
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler = metadata['scaler']
        self.label_encoder = metadata['label_encoder']
        self.class_names = metadata['class_names']
        self.model_type = metadata['model_type']
        
        # Load model
        if self.model_type == "neural_network":
            self.model = keras.models.load_model(model_path)
        
        print(f"Model loaded from: {model_path}")
        print(f"Model type: {self.model_type}")
        print(f"Classes: {', '.join(self.class_names)}")


def main():
    """Example training pipeline."""
    print("=" * 60)
    print("ASL Gesture Recognition - Model Training")
    print("=" * 60)
    
    # Configuration
    DATA_FILE = "data/gestures/gesture_data.csv"  # Update this path
    MODEL_TYPE = "neural_network"  # TensorFlow/Keras model for mobile deployment
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"\nError: Data file not found: {DATA_FILE}")
        print("Please run collect_data.py first to collect training data.")
        return
    
    # Initialize classifier
    classifier = GestureClassifier(model_type=MODEL_TYPE)
    
    # Load data
    X, y = classifier.load_data(DATA_FILE, file_format="csv")
    
    # Preprocess and split data
    X_train, X_test, y_train, y_test = classifier.preprocess_data(X, y, test_size=0.2)
    
    # Create and train model
    classifier.create_model()
    classifier.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    accuracy, y_pred = classifier.evaluate(X_test, y_test)
    
    # Save model
    model_path = classifier.save_model()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model type: {MODEL_TYPE}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Model saved: {model_path}")
    print("\nNext step: Run convert_to_tflite.py to create a .tflite model")


if __name__ == "__main__":
    main()
