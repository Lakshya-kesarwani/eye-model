import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

interface RulerOverlayProps {
  width: number;
  height: number;
}

export const RulerOverlay: React.FC<RulerOverlayProps> = ({ width, height }) => {
  return (
    <>
      {/* Horizontal ruler */}
      <View style={styles.horizontalRulerContainer}>
        {Array.from({ length: Math.ceil(width / 30) + 1 }, (_, i) => (
          <Text
            key={`h-ruler-${i}`}
            style={[
              styles.rulerLabel,
              { left: i * 30 - 5 },
            ]}
          >
            {i * 30}
          </Text>
        ))}
      </View>
      
      {/* Vertical ruler */}
      <View style={styles.verticalRulerContainer}>
        {Array.from({ length: Math.ceil(height / 20) + 1 }, (_, i) => (
          <Text
            key={`v-ruler-${i}`}
            style={[
              styles.rulerLabel,
              { 
                top: i * 20, 
                left: 10,
              },
            ]}
          >
            {i * 20}
          </Text>
        ))}
      </View>
    </>
  );
};

const styles = StyleSheet.create({
  horizontalRulerContainer: {
    position: 'absolute',
    width: '100%',
    height: 15,
    flexDirection: 'row',
    alignItems: 'center',
  },
  verticalRulerContainer: {
    position: 'absolute',
    left: -5,
    height: '100%',
    width: 20,
    top:-2,
    justifyContent: 'space-evenly',
  },
  rulerLabel: {
    position: 'absolute',
    color: 'orange',
    fontSize: 8,
    fontWeight: '900',
  },
});