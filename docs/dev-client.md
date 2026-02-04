# Dev client (EAS) — Android-focused quick steps 🔧

- Install EAS CLI: `npm install -g eas-cli`
- Login: `eas login`
- Add `eas.json` (uses a `development` profile with `developmentClient: true`)
- Install dev client: `expo install expo-dev-client`
- Add native packages (examples): `yarn add react-native-vision-camera react-native-fast-tflite`
- Prebuild: `expo prebuild`
- Build dev client (Android): `eas build --profile development --platform android` → download APK
  - Note: Install the APK on an Android device or emulator. For local emulator testing, use `expo run:android` instead.
- Start with dev client: `expo start --dev-client` (run on device with the installed dev-client app)

Notes:
- Vision Camera requires JSI/Hermes and Android permission config.
- Replace Expo Go on your device with the built dev-client app to test native frame processors.
