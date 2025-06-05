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
    GRID_CONFIG = { rows: 9, cols: 9 }
}: UseCaptureProps) => {
    const [targetIndex, setTargetIndex] = useState(0);
    const [capturing, setCapturing] = useState(true);

    const [frameCount, setFrameCount] = useState(0);
    const cameraRef = useRef<CameraViewRef>(null);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    const dotX = (targetIndex % GRID_CONFIG.cols) * cellWidth + cellWidth / 2 - 2.5;
    const dotY = Math.floor(targetIndex / GRID_CONFIG.cols) * cellHeight + cellHeight / 2 - 2.5;

    const captureAndPredict = async () => {
        if (!cameraRef.current) return;

        try {
            const photo = await cameraRef.current.takePictureAsync({
                quality: 0.2,
                skipProcessing: true,
            });

            const base64 = await FileSystem.readAsStringAsync(photo.uri, {
                encoding: FileSystem.EncodingType.Base64,
            });

            const response = await axios.post('https://bf20-14-139-98-164.ngrok-free.app/predict', {
                image: base64,
            });
            console.log(response.data)
            if (response.data && typeof response.data.predictedIndex === 'number') {
                setTargetIndex(response.data.predictedIndex);
            }
        } catch (err) {
            console.error(err);
        }
    };

    useEffect(() => {
        if (capturing) {
            intervalRef.current = setInterval(() => {
                captureAndPredict();
            }, captureIntervalMs);
        } else {
            if (intervalRef.current) clearInterval(intervalRef.current);
        }
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [capturing]);

    return {
        cameraRef,
        targetIndex,
        capturing,
        frameCount,
        setCapturing,
    };
};