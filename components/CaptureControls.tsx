import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

interface CaptureControlsProps {
  capturing: boolean;
  targetIndex: number;
  totalCells: number;
  frameCount: number;
  framesPerTarget: number;
  onStartCapture: () => void;
}

export const CaptureControls: React.FC<CaptureControlsProps> = ({
  capturing,
  targetIndex,
  totalCells,
  frameCount,
  framesPerTarget,
  onStartCapture,
}) => {
  return (
    <View style={styles.controls}>
      <TouchableOpacity
        onPress={onStartCapture}
        disabled={capturing}
        style={styles.captureButton}
      >
        <Text style={styles.captureText}>
          {capturing ? 'Capturing...' : 'Start Capture'}
        </Text>
      </TouchableOpacity>
      <Text style={styles.status}>
        Cell: {targetIndex + 1}/{totalCells} | Frame: {frameCount}/{framesPerTarget}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  controls: {
    position: 'absolute',
    bottom: 30,
    alignSelf: 'center',
    alignItems: 'center',
  },
  captureButton: {
    backgroundColor: '#1e90ff',
    paddingHorizontal: 30,
    paddingVertical: 10,
    borderRadius: 10,
  },
  captureText: { 
    color: 'white', 
    fontSize: 16 
  },
  status: {
    color: 'white',
    fontSize: 14,
    marginTop: 8,
  },
});