"""
TensorFlow Lite Model Conversion and Optimization

Converts trained models to optimized TFLite format for mobile deployment.
Supports quantization and optimization for reduced model size and faster inference.
"""

import tensorflow as tf
import numpy as np
import pickle
import os
from datetime import datetime


class TFLiteConverter:
    def __init__(self):
        """Initialize TFLite converter."""
        self.converter = None
        self.tflite_model = None
        
    def convert_keras_model(self, model_path, optimize=True, quantize=False):
        """
        Convert Keras/TensorFlow model to TFLite.
        
        Args:
            model_path: Path to .h5 model file
            optimize: Apply default optimizations
            quantize: Apply dynamic range quantization
            
        Returns:
            tflite_model: Converted TFLite model bytes
        """
        print(f"\nLoading Keras model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        # Create converter
        self.converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply optimizations
        if optimize:
            print("Applying default optimizations...")
            self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Apply quantization
        if quantize:
            print("Applying dynamic range quantization...")
            self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
            self.converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        print("Converting to TFLite format...")
        self.tflite_model = self.converter.convert()
        
        print("Conversion successful!")
        return self.tflite_model
    
    def save_tflite_model(self, output_path=None):
        """Save TFLite model to file."""
        if self.tflite_model is None:
            raise ValueError("No TFLite model to save. Run convert_* method first.")
        
        os.makedirs('models', exist_ok=True)
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"models/gesture_model_{timestamp}.tflite"
        
        # Save model
        with open(output_path, 'wb') as f:
            f.write(self.tflite_model)
        
        # Get model size
        model_size = len(self.tflite_model) / 1024  # KB
        
        print(f"\nTFLite model saved to: {output_path}")
        print(f"Model size: {model_size:.2f} KB")
        
        return output_path
    
    def test_tflite_model(self, tflite_path, test_data, test_labels):
        """
        Test TFLite model accuracy.
        
        Args:
            tflite_path: Path to .tflite model
            test_data: Test input data (numpy array)
            test_labels: True labels
        """
        print(f"\nTesting TFLite model: {tflite_path}")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")
        
        # Run inference on test data
        predictions = []
        for sample in test_data:
            interpreter.set_tensor(input_details[0]['index'], 
                                  sample.reshape(1, -1).astype(np.float32))
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(np.argmax(output))
        
        # Calculate accuracy
        predictions = np.array(predictions)
        accuracy = np.mean(predictions == test_labels)
        
        print(f"\nTFLite Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy
    
    def benchmark_model(self, tflite_path, num_runs=100):
        """
        Benchmark TFLite model inference speed.
        
        Args:
            tflite_path: Path to .tflite model
            num_runs: Number of inference runs
        """
        import time
        
        print(f"\nBenchmarking model: {tflite_path}")
        print(f"Running {num_runs} inferences...")
        
        # Load model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create random input
        input_shape = input_details[0]['shape']
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = (total_time / num_runs) * 1000  # ms
        fps = num_runs / total_time
        
        print(f"\nAverage inference time: {avg_time:.2f} ms")
        print(f"Throughput: {fps:.2f} FPS")
        
        return avg_time, fps


def main():
    """Main conversion pipeline."""
    print("=" * 60)
    print("TensorFlow Lite Model Conversion")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = "models/neural_network_latest.h5"  # Update this

    converter = TFLiteConverter()
    
    # Check which model exists
    if os.path.exists(MODEL_PATH):
        print(f"\nConverting TensorFlow/Keras model...")
        
        # Convert with optimization and quantization
        tflite_model = converter.convert_keras_model(
            MODEL_PATH,
            optimize=True,
            quantize=True  # Set to False for no quantization
        )
        
        # Save
        output_path = converter.save_tflite_model("models/gesture_model_optimized.tflite")
        
        # Benchmark
        converter.benchmark_model(output_path, num_runs=100)
    else:
        print("\nError: No trained model found!")
        print("Please train a model first using train_model.py")
        print(f"Looking for:")
        print(f"  - {MODEL_PATH} (TensorFlow/Keras)")
        return
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"TFLite model ready for mobile deployment")
    print(f"Location: {output_path}")
    print(f"\nYou can now use this .tflite model in your React Native app!")


if __name__ == "__main__":
    main()
