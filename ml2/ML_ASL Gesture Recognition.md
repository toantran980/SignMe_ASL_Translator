# ASL Gesture Recognition - ML Pipeline

Complete ML pipeline for training ASL gesture recognition using MediaPipe hand landmarks.

## Quick Start

### 1. Install Dependencies

```bash
cd ml
pip install -r requirements.txt
python download_model.py  # Download MediaPipe hand landmarker model
```

**Requirements:**

- Python 3.10+
- Webcam
- MediaPipe 0.10.32 (uses task-based API)

### 2. Collect Training Data

```bash
python collect_data.py
```

**Menu Options:**

- **0:** Test camera/hand detection (preview mode - use this first!)
- **1:** List predefined gestures
- **2:** Collect single gesture
- **3:** Collect all gestures (8 predefined)
- **4:** Save and exit
- **5:** Exit without saving

**Recording Controls:**

- `SPACE` - Start/resume recording
- `P` - Pause
- `Q` - Save
- `ESC` - Discard

**Tips:**

- Start with option 0 to verify hand detection works
- Record 100-200 samples per gesture
- Good lighting improves accuracy
- Vary hand positions and angles

### 3. Train Model

```bash
python train_model.py
```

**Supported Models:**

- Random Forest (default, fast & accurate)
- SVM, KNN, Gradient Boosting
- Neural Network (TensorFlow)

### 4. Convert to TFLite

```bash
python convert_to_tflite.py
```

Generates optimized `.tflite` model for mobile deployment.

## 📁 Files

```
ml/
├── collect_data.py          # Data collection (MediaPipe 0.10.32 task API)
├── train_model.py            # Model training
├── convert_to_tflite.py      # TFLite conversion
├── download_model.py         # Download hand_landmarker.task
├── requirements.txt          # Dependencies
├── hand_landmarker.task      # MediaPipe model (auto-downloaded)
└── data/gestures/            # Training data (auto-created)
```

## 🎯 Workflow Details

### Data Collection (MediaPipe 0.10.32)

Extracts 21 hand landmarks (x,y,z) = 63 features per sample.

**Output:**

- `gesture_data_TIMESTAMP.csv` - CSV with 63 features + label
- `gesture_data_TIMESTAMP.npz` - NumPy compressed format

### Model Training

- 80/20 train/test split
- Feature scaling with StandardScaler
- Cross-validation
- Saves confusion matrix

**Model Comparison:**

| Model          | Accuracy | Speed | Size   |
| -------------- | -------- | ----- | ------ |
| Random Forest  | 95-98%   | Fast  | Medium |
| SVM            | 93-96%   | Fast  | Small  |
| Neural Network | 94-98%   | Fast  | Small  |

### TFLite Conversion

- Default optimization (~50% size reduction)
- Optional quantization (~75% reduction)
- Benchmarks inference speed

## 🎓 Recommended Gestures

Start with 5-8 distinct gestures:

- **Greetings:** hello, goodbye, thank_you
- **Common:** please, yes, no, help, stop

## 🔧 Troubleshooting

**Low accuracy (<85%):**

- Collect more samples (150-200 per gesture)
- Try Neural Network or Gradient Boosting
- Ensure gestures are visually distinct

**Camera not detected:**

- Check webcam permissions
- Try `cv2.VideoCapture(1)` for external webcam

**Model too large:**

- Use quantization
- Reduce neural network size

## 🔄 Deploy to App

```bash
# Copy model to app assets
cp models/gesture_model.tflite ../assets/models/
```

Update label mapping in your React Native code.

## 📚 Resources

- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [TensorFlow Lite](https://www.tensorflow.org/lite/guide)
- [ASL Reference](https://www.lifeprint.com/asl101/pages-layout/concepts.htm)
