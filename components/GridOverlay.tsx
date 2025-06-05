import React from 'react';
import { View, StyleSheet } from 'react-native';
import { TouchableOpacity, Text, TextInput } from 'react-native';
import { getCluster, CLUSTER_CONFIG } from '../utils/gridutils';
interface GridOverlayProps {
  width: number;
  height: number;
  rows: number;
  cols: number;
  isApp2?: boolean; // Optional prop to indicate if it's App2
  targetIndex?: number; // Optional prop for target index, replace with actual logic if needed
}
const BUTTONS = [
  { key: 'A', row: 0, col: 0 }, // Top-left
  { key: 'B', row: 0, col: 3 }, // Top-center
  { key: 'C', row: 0, col: 6 }, // Top-right
  { key: 'D', row: 3, col: 0 }, // Middle-left
  { key: 'E', row: 3, col: 6 }, // Middle-right
  { key: 'F', row: 6, col: 0 }, // Bottom-left
  { key: 'G', row: 6, col: 3 }, // Bottom-center
  { key: 'H', row: 6, col: 6 }, // Bottom-right
];

export const GridOverlay: React.FC<GridOverlayProps> = ({
  width,
  height,
  rows,
  cols,
  isApp2 = false, // Default to false if not provided
  targetIndex = 0, // Default to 0 if not provided
}) => {
  const cellWidth = width / cols;
  const cellHeight = height / rows;
  const [text, setText] = React.useState('');
  const cluster = getCluster(targetIndex);
  React.useEffect(() => {
    const label = CLUSTER_CONFIG.labels[cluster];
    if (!isApp2) return;
    if (label === 'BKL') {
      setText((prev) => prev.slice(0, -1)); // Backspace
    } else if (label) {
      setText((prev) => prev + label); // Append label
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetIndex]);

  console.log(`Target index: ${targetIndex}, Cluster: ${CLUSTER_CONFIG.labels[cluster]}`);
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

      {isApp2 && (
        <> {/* 8 Buttons overlay */}
          {BUTTONS.map(btn => (
            <TouchableOpacity
              key={btn.key}
              style={[
                styles.overlayButton,
                {
                  left: btn.col * cellWidth,
                  top: btn.row * cellHeight,
                  width: cellWidth * 3,
                  height: cellHeight * 3,
                },
              ]}
              onPress={() => {
                // Handle button press
                console.log(`Button ${btn.key} pressed`);
                setText(text + btn.key);
              }}
            >
              <Text style={styles.buttonText}>{btn.key}</Text>
            </TouchableOpacity>
          ))}

          {/* Center 3x3 text bar */}
          <View
            style={[
              styles.centerTextBar,
              {
                left: cellWidth * 3,
                top: cellHeight * 3,
                width: cellWidth * 3,
                height: cellHeight * 3,
              },
            ]}
          >
            <TextInput
              style={styles.textInput}
              placeholder="Enter text"
              placeholderTextColor="#ccc"
              value={text}
              onChangeText={(text) => setText(text)}
              multiline={true}
            />
          </View>
        </>
      )}

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

  overlayButton: {
    position: 'absolute',
    backgroundColor: 'rgba(0,0,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 5,
    borderRadius: 8,
  },
  buttonText: {
    fontSize: 28,
    color: '#fff',
    fontWeight: 'bold',
  },
  centerTextBar: {
    position: 'absolute',
    backgroundColor: 'rgba(100,0,0,0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1,
    borderRadius: 8,
    padding: 1,

  },
  textInput: {
    width: '90%',
    height: 250,
    backgroundColor: 'rgba(100,0,100,0.4)',
    borderRadius: 6,
    paddingHorizontal: 5,
    fontSize: 18,
    color: '#fff',
    fontWeight: 'bold',
    textAlign: 'left',
    textAlignVertical: 'top', // Align text to the top
    // wrap
    flexWrap: 'wrap',
  },
});