# SignMe ASL Translator

Real-time ASL gesture recognition mobile app using MediaPipe hand tracking and TensorFlow Lite.

## Quick Start

### 1. Install Dependencies

```bash
npm install --legacy-peer-deps
npx expo install --check
```

### 2. Run Metro (Docker or Local)

**Docker (Recommended):**

```bash
docker compose up --build
```

**Clean up Docker:**

```docker-compose
docker-compose down -v
```

**Local:**

```bash
npm start
```

### 3. ML Pipeline (Python)

See [ml/README.md](ml/README.md) for complete ML workflow.

```bash
cd ml
pip install -r requirements.txt
python collect_data.py    # Collect gesture data
python train_model.py     # Train classifier
python convert_to_tflite.py  # Export .tflite
```

## Development Modes

**Expo Go (UI Testing):**

```bash
npm start
# Scan QR with Expo Go app
```

**Dev Client (Full Features):**

```bash
# Install EAS CLI
npm install -g eas-cli
eas login

# Build dev client for Android
expo prebuild
eas build --profile development --platform android

# Or local emulator
expo run:android

# Start
expo start --dev-client
```

## Android MediaPipe Integration

See [docs/android-mediapipe.md](docs/android-mediapipe.md) for native frame processor setup.

**Key Steps:**

1. Add MediaPipe dependency to `android/app/build.gradle`
2. Implement Kotlin frame processor plugin for Vision Camera
3. Extract hand landmarks (21×3 = 63 features)
4. Pass to TFLite model for gesture classification

## Architecture

- **Mobile App:** React Native + Expo (standalone, runs on-device)
- **ML Pipeline:** Python scripts (offline training only)
- **Model:** TensorFlow Lite (.tflite file embedded in app)
- **Hand Tracking:** MediaPipe Hands (native Android plugin)
