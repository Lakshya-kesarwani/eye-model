// --jsx
import React, { useEffect, useRef, useState } from 'react';
import { View, Button, StyleSheet, PermissionsAndroid, Platform } from 'react-native';
import { Camera, useCameraDevice, VideoFile } from 'react-native-vision-camera';
import type { Camera as CameraType } from 'react-native-vision-camera';
import { jsx } from 'react/jsx-runtime';

const CameraRecorder: React.FC = () => {
  const camera = useRef<CameraType>(null);
  const device = useCameraDevice('front');

  const [hasPermission, setHasPermission] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);

  useEffect(() => {
    const requestPermissions = async () => {
      const cameraPermission = await Camera.requestCameraPermission();
      const micPermission = await Camera.requestMicrophonePermission();

      if (
        cameraPermission === Camera.PermissionStatus.AUTHORIZED &&
        micPermission === Camera.PermissionStatus.AUTHORIZED
      ) {
        setHasPermission(true);
      }
    };

    requestPermissions();
  }, []);

  const startRecording = async () => {
    if (!camera.current) return;

    setIsRecording(true);
    setIsPaused(false);

    await camera.current.startRecording({
      onRecordingFinished: (video: VideoFile) => {
        console.log('Recording finished:', video.path);
      },
      onRecordingError: (error: Error) => {
        console.error('Recording error:', error);
      },
    });
  };

  const stopRecording = async () => {
    if (!camera.current) return;

    await camera.current.stopRecording();
    setIsRecording(false);
    setIsPaused(false);
  };

  const pauseRecording = async () => {
    if (!camera.current || !isRecording || isPaused) return;

    await camera.current.pauseRecording();
    setIsPaused(true);
  };

  const resumeRecording = async () => {
    if (!camera.current || !isRecording || !isPaused) return;

    await camera.current.resumeRecording();
    setIsPaused(false);
  };

  if (!device || !hasPermission) {
    return (
      <View style={styles.centered}>
        <Button title="Requesting camera permission..." disabled />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        video={true}
        audio={true}
      />
      <View style={styles.controls}>
        {!isRecording ? (
          <Button title="Start Recording" onPress={startRecording} />
        ) : (
          <>
            <Button title="Stop" onPress={stopRecording} />
            {isPaused ? (
              <Button title="Resume" onPress={resumeRecording} />
            ) : (
              <Button title="Pause" onPress={pauseRecording} />
            )}
          </>
        )}
      </View>
    </View>
  );
};

export default CameraRecorder;

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  controls: {
    position: 'absolute',
    bottom: 40,
    left: 20,
    right: 20,
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
});
