import { TouchableOpacity, Text, StyleSheet, ViewStyle } from 'react-native';
import React from 'react'
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

interface ButtonProps {
    label: string;
    style?: ViewStyle;
    onPress?: () => void;
    cellWidth?: number;
    cellHeight?: number;
}

const Button: React.FC<ButtonProps> = ({ label, style, onPress,cellWidth,cellHeight }) => (
    <TouchableOpacity style={[styles.button, style]} onPress={onPress}>
        <Text style={styles.text}>{label}</Text>
    </TouchableOpacity>
);
export default Button

const styles = StyleSheet.create({
    button: {
        backgroundColor: 'rgba(0,0,255,0.2)',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 8,
        position: 'absolute',
        zIndex: 5,
    },
    text: {
        fontSize: 28,
        color: '#fff',
        fontWeight: 'bold',
    },
});