import React,{useState,useEffect} from 'react';
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
  if(dotY>260 && dotY<400 ){
    return <View style={[styles.dot, { left:dotX, top:dotY*1.1 }]} />;
  }
  else if(dotY>440 ){
  return <View style={[styles.dot, { left:dotX, top:dotY*0.7+250 }]} />;}
  return <View style={[styles.dot, { left: dotX, top: dotY }]} />;
};

const styles = StyleSheet.create({
  dot: {
    position: 'absolute',
    width: 15,
    height: 15,
    borderRadius: 10,
    backgroundColor: 'red',
    zIndex: 5,
    opacity: 0.5,
  },
});