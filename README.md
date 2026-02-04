# SignMe ASL Translator

## Setup Commands

### Initial Setup
```bash
# Install dependencies with legacy peer deps support
npm install --legacy-peer-deps

# Clean reinstall if needed
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps
```

### Docker Setup (Recommended)

**Deep Clean (if issues)**
```bash
docker-compose down --rmi all --volumes --remove-orphans
docker compose up --build
```

**Close and Clean**
```bash
docker-compose down -v
```

**Start Fresh**
```bash
docker-compose up --build
```

**Accessing the App**
- Scan the QR code from Expo tunnel
- App runs on port 8081

---

## Dev client (EAS) — Android (short steps) 🔧

- Install EAS CLI and login: `npm install -g eas-cli` then `eas login`
- Add `eas.json` with a `development` profile and set `developmentClient: true`
- Install dev client: `expo install expo-dev-client`
- Add native deps (example): `yarn add react-native-vision-camera react-native-fast-tflite react-native-reanimated react-native-permissions`
- Prebuild & build dev client (Android): `expo prebuild` then `eas build --profile development --platform android`
- Start with the dev client: `expo start --dev-client` and open the built app on your Android device

> See `docs/dev-client.md` for more details and `docs/android-mediapipe.md` for Android MediaPipe integration notes.

## Python — data collection & model (short steps) 

- Create & activate venv (PowerShell): `python -m venv .venv` then `\.venv\Scripts\Activate.ps1`
- Install libs: `pip install mediapipe opencv-python numpy pandas tensorflow scikit-learn`
- Use `scripts/collect_landmarks.py` to record per-frame landmarks (CSV/NumPy)
- Train in TensorFlow/Keras and export a quantized `.tflite` for on-device use

## Running (quick)

- **Docker (local dev)**: `docker compose up --build` — Metro runs on port 8081 (tunnel mode used by default).
- **Expo Go (UI-only)**: `expo start` → open with Expo Go (fast iteration; native frame-processor/TFLite won't run).
- **Dev Client (native testing)**: `expo prebuild` → `eas build --profile development --platform ios` → install built dev-client on device → `expo start --dev-client`.
