# Project Checklist — SignMe ASL Translator ✅

_Last updated: 2026-02-03_

---

## Quick status

- Overall: **Frontend focused & dev-client ready (iOS)**. ✅
- Docker: Metro runs via Docker. ✅
- Native prebuild: `ios/` and `android/` generated. ✅

---

## Done ✅

- Project & dev tools
  - `eas.json` added (development profile). ✅
  - `app.json` updated: **Hermes enabled** and valid **bundleIdentifier** (`com.signme.asl`). ✅
  - `expo prebuild` executed → native projects created. ✅
- Packages installed
  - Installed: `expo-dev-client`, `react-native-vision-camera`, `react-native-fast-tflite`, `react-native-reanimated`, `react-native-permissions`. ✅
- Repo scaffolding
  - `app/screens/CameraScreen.jsx` (JS skeleton + mock-ready) ✅
  - `utils/tflite.js` (TFLite stub) ✅
  - `docs/ios-mediapipe.md`, `docs/dev-client.md`, `README.md` updated ✅
- Docker
  - `Dockerfile` and `docker-compose.yml` fixed for reliability (bind & install). ✅

---

## High-priority next steps (short-term) 

- Mobile / Native
  - [ ] Finish iOS dev-client build on EAS: `eas build --profile development --platform ios` (requires Apple Developer access or allow EAS to manage credentials). **Owner:** Mobile
  - [ ] Implement iOS frame-processor plugin (Swift/ObjC) to run MediaPipe Hand Landmarker and return flattened landmarks [x1,y1,z1,...]. **Owner:** Mobile
  - [ ] Integrate and test `.tflite` on device via `react-native-fast-tflite` or native TFLite wrapper. **Owner:** Mobile / ML
- Python / ML
  - [ ] Collect landmark data using MediaPipe Python and save (CSV/NumPy). **Owner:** Python
  - [ ] Train model (TF/Keras), quantize & export `.tflite`. **Owner:** Python
- UI / Integration
  - [ ] Add mock-landmark provider for Expo Go so UI team can iterate without native plugin. **Owner:** UI
  - [ ] Hook landmarks → normalization → TFLite inference flow in `CameraScreen.jsx`. **Owner:** UI/Mobile

---

## Nice-to-have 

- [ ] Add `expo-system-ui` if you want `userInterfaceStyle` support: `expo install expo-system-ui`.
- [ ] Add tests for model output & integration.
- [ ] Add a sample `.tflite` test resource and CI check for model inference.

---

## Dev / Run commands (cheat sheet) 🧾

- Docker: `docker compose up --build` (Metro on :8081)
- Expo Go (UI): `expo start` → open with Expo Go (no native plugins)
- Dev client (iOS): `expo prebuild` → `eas build --profile development --platform ios` → install dev-client → `expo start --dev-client`
- Local Android run: `expo run:android` or `npx react-native run-android`


If you want, I can:

- scaffold the **mock-landmark provider** in `CameraScreen.jsx` (fast)
- or start the **iOS EAS build** and help with credential prompts (requires your decision on credential handling). 

Which should I do next? 
