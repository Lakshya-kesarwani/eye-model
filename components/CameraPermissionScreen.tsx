

// components/CameraPermissionScreen.tsx
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

interface CameraPermissionScreenProps {
  onRequestPermission: () => void;
}

export const CameraPermissionScreen: React.FC<CameraPermissionScreenProps> = ({
  onRequestPermission,
}) => {
  return (
    <View style={styles.centered}>
      <Text style={styles.text}>Camera permission needed.</Text>
      <TouchableOpacity onPress={onRequestPermission}>
        <Text style={styles.button}>Grant Access</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  centered: { 
    flex: 1, 
    justifyContent: 'center', 
    alignItems: 'center' 
  },
  text: { 
    color: 'white', 
    fontSize: 16 
  },
  button: { 
    fontSize: 18, 
    padding: 10, 
    color: 'skyblue' 
  },
});