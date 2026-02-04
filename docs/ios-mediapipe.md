# iOS — MediaPipe integration notes (short)

- Goal: run MediaPipe Hand Landmarker on each camera frame and return 21×(x,y,z) landmarks to JS.

Quick steps (high level):
1. Add MediaPipe iOS dependency (framework or CocoaPod). See MediaPipe iOS docs: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/ios
2. Add TensorFlow Lite iOS (e.g., `pod 'TensorFlowLiteSwift'`) to run `.tflite` on-device.
3. Implement an iOS native module / frame-processor plugin for `react-native-vision-camera`:
   - Follow Vision Camera "Write a Frame Processor Plugin" guide (uses JSI/worklets).
   - The plugin receives a frame buffer, runs MediaPipe hand landmarker (C++/ObjC/Swift), and returns a JS-serializable array of floats: `[x1,y1,z1,...,x21,y21,z21]`.
4. From the frame-processor worklet, call `runOnJS` to forward the landmarks to the JS side (see `app/screens/CameraScreen.jsx`).
5. On JS side, normalize landmarks (center and scale) and call your TFLite model binding (e.g., `react-native-fast-tflite` or a custom native TFLite wrapper) to get the gesture label.

Notes:
- iOS Simulator may not provide a real camera feed; test on a device for accurate results.
- Building the dev client for iOS requires an Apple account / credentials for EAS builds.
- If you prefer, we can scaffold a minimal Xcode Swift plugin skeleton that receives frames and returns a dummy landmark array to help iterate on the RN side while native code is being implemented.
