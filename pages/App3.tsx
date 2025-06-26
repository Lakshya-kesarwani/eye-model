import { StyleSheet, Text, View } from 'react-native'
import React from 'react'



export default function App3(){
    return (
        <View style={styles.container}>
        <Text style={styles.title}>App3</Text>
        <Text style={styles.description}>This is a placeholder for App3 functionality.</Text>
        </View>
    )
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#f0f8ff',
    },
    title: {
        fontSize: 28,
        marginBottom: 20,
        fontWeight: 'bold',
    },
    description: {
        fontSize: 18,
        color: '#333',
        textAlign: 'center',
        paddingHorizontal: 20,
    },
})