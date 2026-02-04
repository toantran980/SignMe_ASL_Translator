# MediaPipe integration notes (short)

- Goal: run MediaPipe Hand Landmarker on each camera frame and return 21×(x,y,z) landmarks to JS.

Quick steps (high level):

1. Add MediaPipe dependency (Gradle). See MediaPipe docs: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/android
2. Add TensorFlow Lite (e.g., `implementation 'org.tensorflow:tensorflow-lite:2.14.0'`) to run `.tflite` on-device.
3. Implement a native module / frame-processor plugin for `react-native-vision-camera`:
   - Follow Vision Camera "Write a Frame Processor Plugin" guide (uses JSI/worklets).
   - The plugin receives a frame buffer, runs MediaPipe hand landmarker (Kotlin/Java), and returns a JS-serializable array of floats: `[x1,y1,z1,...,x21,y21,z21]`.
4. From the frame-processor worklet, call `runOnJS` to forward the landmarks to the JS side (see `app/screens/CameraScreen.jsx`).
5. On JS side, normalize landmarks (center and scale) and call your TFLite model binding (e.g., `react-native-fast-tflite` or a custom native TFLite wrapper) to get the gesture label.

Notes:

- Android Emulator may not provide a real camera feed; test on a physical device for accurate results.
- Building the dev client for Android requires a Google Play Console account / credentials for EAS builds, or you can generate a local APK.
- If you prefer, we can scaffold a minimal Kotlin plugin skeleton that receives frames and returns a dummy landmark array to help iterate on the RN side while native code is being implemented.
