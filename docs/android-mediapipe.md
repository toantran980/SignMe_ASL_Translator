# Android MediaPipe Integration

Integrate MediaPipe Hand Landmarker with React Native Vision Camera for real-time gesture detection.

## Goal

Run MediaPipe on each camera frame and return 21×(x,y,z) = 63 features to JavaScript for TFLite classification.

## Setup Steps

### 1. Add Dependencies

**In `android/app/build.gradle`:**
```gradle
dependencies {
    // MediaPipe Tasks Vision
    implementation 'com.google.mediapipe:tasks-vision:0.10.14'
    
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
}
```

### 2. Download Model

Download `hand_landmarker.task` from MediaPipe:
```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Place in `android/app/src/main/assets/`

### 3. Create Frame Processor Plugin

**File:** `android/app/src/main/java/com/ttran286/SignMe_ASL_Translator/HandLandmarkerPlugin.kt`

```kotlin
package com.ttran286.SignMe_ASL_Translator

import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.mrousavy.camera.frameprocessor.Frame
import com.mrousavy.camera.frameprocessor.FrameProcessorPlugin

class HandLandmarkerPlugin : FrameProcessorPlugin() {
    private lateinit var handLandmarker: HandLandmarker
    
    override fun callback(frame: Frame, arguments: Map<String, Any>?): Any? {
        // Convert frame to MediaPipe Image
        val mpImage = convertFrameToMPImage(frame)
        
        // Run detection
        val result = handLandmarker.detect(mpImage)
        
        // Return landmarks as flat array [x1,y1,z1,...,x21,y21,z21]
        return result.landmarks().firstOrNull()?.let { hand ->
            hand.flatMap { listOf(it.x(), it.y(), it.z()) }
        } ?: emptyList<Float>()
    }
    
    // Initialize in constructor
    // Implementation details omitted for brevity
}
```

### 4. Use in React Native

**In `app/screens/CameraScreen.jsx`:**
```javascript
import { useFrameProcessor } from 'react-native-vision-camera';
import { runOnJS } from 'react-native-reanimated';

const frameProcessor = useFrameProcessor((frame) => {
  'worklet';
  
  // Get landmarks from native plugin
  const landmarks = __detectHandLandmarks(frame);
  
  if (landmarks && landmarks.length === 63) {
    // Forward to JS thread for TFLite inference
    runOnJS(processLandmarks)(landmarks);
  }
}, []);

function processLandmarks(landmarks) {
  // Normalize & run TFLite model
  const prediction = model.predict(landmarks);
  console.log('Gesture:', prediction);
}
```

## Notes

- **Emulator:** Limited camera support; test on physical device
- **Performance:** MediaPipe runs at 30+ fps on modern devices
- **Permissions:** Add camera permissions in `AndroidManifest.xml`

## Resources

- [MediaPipe Hand Landmarker Android](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/android)
- [Vision Camera Frame Processors](https://react-native-vision-camera.com/docs/guides/frame-processors)
