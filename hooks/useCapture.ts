
// hooks/useCapture.ts
import { useRef, useEffect, useState } from 'react';
import axios from 'axios';
import type { CameraViewRef } from 'expo-camera';
import * as FileSystem from 'expo-file-system';

interface UseCaptureProps {
    captureIntervalMs: number;
    framesPerTarget: number;
    totalCells: number;
    cellWidth?: number;
    cellHeight?: number;
    GRID_CONFIG?: {
        rows: number;
        cols: number;
    }
}

export const useCapture = ({
    captureIntervalMs,
    framesPerTarget,
    totalCells,
    cellWidth,
    cellHeight,
    GRID_CONFIG = { rows: 9, cols: 9}
}: UseCaptureProps) => {
    const [targetIndex, setTargetIndex] = useState(0);
    const [capturing, setCapturing] = useState(false);
    const [frameCount, setFrameCount] = useState(0);
    const cameraRef = useRef<CameraViewRef>(null);

    useEffect(() => {
        if (capturing) startCapture();
    }, [capturing]);
    const dotX = (targetIndex % GRID_CONFIG.cols) * cellWidth + cellWidth / 2 - 2.5;
    const dotY = Math.floor(targetIndex / GRID_CONFIG.cols) * cellHeight + cellHeight / 2 - 2.5;
    const startCapture = () => {
        if (!cameraRef.current) return;

        let captured = 0;
        const interval = setInterval(async () => {
            if (!cameraRef.current || captured >= framesPerTarget) {
                clearInterval(interval);
                setCapturing(false);
                setFrameCount(0);
                setTargetIndex((targetIndex + 1) % totalCells);
                return;
            }

            try {
                const photo = await cameraRef.current.takePictureAsync({
                    quality: 0.3,
                    skipProcessing: true,
                });

                const base64 = await FileSystem.readAsStringAsync(photo.uri, {
                    encoding: FileSystem.EncodingType.Base64,
                });

                const label = `cell-${targetIndex}`;
                await axios.post('https://8e55-14-139-98-164.ngrok-free.app/upload', {
                    image: base64,
                    x: dotX,
                    y: dotY,
                    index: targetIndex,
                });

                captured++;
                setFrameCount(captured);
            } catch (err) {
                console.error(err);
                clearInterval(interval);
                setCapturing(false);
            }
        }, captureIntervalMs);
    };

    return {
        cameraRef,
        targetIndex,
        capturing,
        frameCount,
        setCapturing,
    };
};