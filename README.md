# SignMe ASL Translator

Set up EXPO (without docker):

npx create-expo-app@latest --template blank ./

Download Expo Go

npx expo start

Install and edit conf follow guide (Install Expo Router ): [https://docs.expo.dev/router/installation/#modify-project-configuration](https://docs.expo.dev/router/installation/#modify-project-configuration)

**Docker (use this)**

Deep Clean if old conf.

```
docker-compose down --rmi all --volumes --remove-orphansdocker compose up --build
```

Close and clean docker instance:

```
docker-compose down -v
```

Start and Rebuild Fresh

```
docker-compose up --build
```

* Note: remove --build if no change

Accessing the App:

* Scan the QR code

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
