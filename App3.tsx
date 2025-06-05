import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import App from './App';
import App2 from './App2';

const App3 = () => {
  const [mode, setMode] = useState<'app1' | 'app2' | null>(null);

  if (mode === 'app1') return <App />;
  if (mode === 'app2') return <App2 />;

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Select App Mode</Text>
      <TouchableOpacity style={styles.button} onPress={() => setMode('app1')}>
        <Text style={styles.buttonText}>Create Dataset</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.button} onPress={() => setMode('app2')}>
        <Text style={styles.buttonText}>Test Application</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  title: { fontSize: 28, marginBottom: 40, fontWeight: 'bold' },
  button: {
    backgroundColor: '#4682B4',
    padding: 20,
    borderRadius: 10,
    marginVertical: 10,
    width: 200,
    alignItems: 'center',
  },
  buttonText: { color: '#fff', fontSize: 20, fontWeight: 'bold' },
});

export default App3;