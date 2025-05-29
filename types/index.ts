export interface CameraPermission {
  granted: boolean;
  canAskAgain: boolean;
  status: string;
}

export interface GridConfig {
  rows: number;
  cols: number;
}

export interface CaptureConfig {
  captureIntervalMs: number;
  framesPerTarget: number;
}

export interface CaptureState {
  targetIndex: number;
  capturing: boolean;
  frameCount: number;
}