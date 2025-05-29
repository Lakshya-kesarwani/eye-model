import React from 'react';
import { View, StyleSheet } from 'react-native';

interface TargetDotProps {
  targetIndex: number;
  cellWidth: number;
  cellHeight: number;
  cols: number;
}

export const TargetDot: React.FC<TargetDotProps> = ({
  targetIndex,
  cellWidth,
  cellHeight,
  cols,
}) => {
  const dotX = (targetIndex % cols) * cellWidth + cellWidth / 2 - 2.5;
  const dotY = Math.floor(targetIndex / cols) * cellHeight + cellHeight / 2 - 2.5;
  // console.log(`TargetDot position: (${dotX}, ${dotY}) for index ${targetIndex}`);
  return <View style={[styles.dot, { left: dotX, top: dotY }]} />;
};

const styles = StyleSheet.create({
  dot: {
    position: 'absolute',
    width: 5,
    height: 5,
    borderRadius: 10,
    backgroundColor: 'red',
    zIndex: 5,
  },
});