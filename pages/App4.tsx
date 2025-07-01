import React, { useState, useEffect } from 'react'
import { StyleSheet ,Button} from 'react-native'
import { Camera, useCameraPermission,useCameraDevice } from 'react-native-vision-camera'

export default function App() {
    const device = useCameraDevice('back')
    const { hasPermission, requestPermission } = useCameraPermission()

    if (!hasPermission) return (
        <Button
            title="Request Camera Permission"
            onPress={requestPermission}
        />
    )
    if (device == null) return (
        <Button
            title="No Camera Device Found"
            onPress={() => console.log('No camera device available')}
        />
    )
    return (
        <Camera
            style={StyleSheet.absoluteFill}
            device={device}
            isActive={true}
        />
    )
}

