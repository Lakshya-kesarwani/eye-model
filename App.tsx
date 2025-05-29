import React, { useEffect } from 'react';
import { CameraView, useCameraPermissions } from 'expo-camera';
import {
  Dimensions,
  StatusBar,
  StyleSheet,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

// Import components
import { CameraPermissionScreen } from './components/CameraPermissionScreen';
import { GridOverlay } from './components/GridOverlay';
import { TargetDot } from './components/TargetDot';
import { RulerOverlay } from './components/RulerOverlay';
import { CaptureControls } from './components/CaptureControls';
import { useCapture } from './hooks/useCapture';

// Constants
const { width, height } = Dimensions.get('screen');
console.log(`Screen dimensions: ${width}x${height}`);

const GRID_CONFIG = {
  rows: 9,
  cols: 9,
};

const CAPTURE_CONFIG = {
  captureIntervalMs: 500, 
  framesPerTarget: 10,
};

const totalCells = GRID_CONFIG.rows * GRID_CONFIG.cols;
const cellWidth = width / GRID_CONFIG.cols;
const cellHeight = height / GRID_CONFIG.rows;
export default function App() {
  const [permission, requestPermission] = useCameraPermissions();

  const {
    cameraRef,
    targetIndex,
    capturing,
    frameCount,
    setCapturing,
  } = useCapture({
    captureIntervalMs: CAPTURE_CONFIG.captureIntervalMs,
    framesPerTarget: CAPTURE_CONFIG.framesPerTarget,
    totalCells,
    cellWidth: cellWidth,
    cellHeight: cellHeight,
    GRID_CONFIG
  });

  useEffect(() => {
    if (!permission) requestPermission();
    StatusBar.setHidden(true);
  }, []);

  if (!permission?.granted) {
    return <CameraPermissionScreen onRequestPermission={requestPermission} />;
  }



  return (
    <SafeAreaView style={styles.container}>
      <CameraView
        style={StyleSheet.absoluteFill}
        ref={cameraRef}
        facing="front"
      />

      <GridOverlay
        width={width}
        height={height}
        rows={GRID_CONFIG.rows}
        cols={GRID_CONFIG.cols}
      />

      <TargetDot
        targetIndex={targetIndex}
        cellWidth={cellWidth}
        cellHeight={cellHeight}
        cols={GRID_CONFIG.cols}
      />

      <RulerOverlay width={width} height={height} />

      <CaptureControls
        capturing={capturing}
        targetIndex={targetIndex}
        totalCells={totalCells}
        frameCount={frameCount}
        framesPerTarget={CAPTURE_CONFIG.framesPerTarget}
        onStartCapture={() => setCapturing(true)}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black'
  },
});