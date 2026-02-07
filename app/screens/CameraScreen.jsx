import React, {useEffect, useState} from 'react'
import {StyleSheet, View, Text, TouchableOpacity} from 'react-native'
import runTFLiteModel from '../../utils/tflite'

// Mock landmarks provider for development (simulates camera + MediaPipe output)  ---> must be modify to fit real implementation
// When native frame-processor is ready, replace this with real Vision Camera integration

export default function CameraScreen(){
  const [label, setLabel] = useState('—')
  const [isRunning, setIsRunning] = useState(false)

  // Simulate receiving landmarks from MediaPipe
  const simulateLandmarks = async () => {
    // Generate random landmarks 
    const mockLandmarks = new Float32Array(63)
    for (let i = 0; i < 63; i++) {
      mockLandmarks[i] = Math.random()
    }
    
    try {
      const result = await runTFLiteModel(mockLandmarks)
      setLabel(result?.label ?? 'no-detect')
    } catch (e) {
      console.warn('TFLite inference failed:', e)
      setLabel('error')
    }
  }

  useEffect(() => {
    let interval
    if (isRunning) {
      interval = setInterval(() => {
        simulateLandmarks()
      }, 500) // Inference every 500ms
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isRunning])

  return (
    <View style={styles.container}>
      {/* Placeholder for camera view — will be replaced with Vision Camera when native is ready */}
      <View style={styles.cameraPlaceholder}>
        <Text style={styles.placeholderText}> Camera (mock)</Text>
      </View>
      
      {/* Prediction display */}
      <View style={styles.overlay}>
        <Text style={styles.label}>Prediction: {label}</Text>
        <Text style={styles.subtitle}>Running: {isRunning ? 'Yes' : 'No'}</Text>
      </View>
      
      {/* Controls */}
      <View style={styles.buttonGroup}>
        <TouchableOpacity 
          style={[styles.button, isRunning && styles.buttonActive]} 
          onPress={() => setIsRunning(!isRunning)}
        >
          <Text style={styles.buttonText}>{isRunning ? 'Stop' : 'Start'}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.button} onPress={() => setLabel('—')}>
          <Text style={styles.buttonText}>Reset</Text>
        </TouchableOpacity>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: '#f5f5f5'},
  cameraPlaceholder: {
    flex: 1,
    backgroundColor: '#ddd',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  placeholderText: {fontSize: 24, fontWeight: '600', color: '#666'},
  overlay: {position: 'absolute', top: 40, left: 20, backgroundColor: 'rgba(0,0,0,0.6)', padding: 10, borderRadius: 8},
  label: {color: '#fff', fontSize: 18, fontWeight: '600'},
  subtitle: {color: '#ccc', fontSize: 12, marginTop: 5},
  buttonGroup: {flexDirection: 'row', gap: 10, padding: 20},
  button: {flex: 1, backgroundColor: '#fff', padding: 12, borderRadius: 8, alignItems: 'center'},
  buttonActive: {backgroundColor: '#4CAF50'},
  buttonText: {color: '#000', fontWeight: '600', fontSize: 16},
})
