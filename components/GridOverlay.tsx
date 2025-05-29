import React from 'react';
import { View, StyleSheet } from 'react-native';

interface GridOverlayProps {
  width: number;
  height: number;
  rows: number;
  cols: number;
}

export const GridOverlay: React.FC<GridOverlayProps> = ({
  width,
  height,
  rows,
  cols,
}) => {
  return (
    <>
      {/* Vertical grid lines */}
      {Array.from({ length: cols - 1 }, (_, i) => (
        <View
          key={`v-${i + 1}`}
          style={[
            styles.gridLine,
            {
              left: (width / cols) * (i + 1),
              height,
              width: 1,
            },
          ]}
        />
      ))}
      {/* Horizontal grid lines */}
      {Array.from({ length: rows - 1 }, (_, i) => (
        <View
          key={`h-${i + 1}`}
          style={[
            styles.gridLine,
            {
              top: (height / rows) * (i + 1),
              width,
              height: 1,
            },
          ]}
        />
      ))}
    </>
  );
};

const styles = StyleSheet.create({
  gridLine: {
    position: 'absolute',
    backgroundColor: 'white',
    opacity: 0.5,
    zIndex: 3,
  },
});