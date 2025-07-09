import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet,Image } from 'react-native';
import App from './pages/App';
import App2 from './pages/App2';

import App3 from './pages/App3'; // Assuming App3 is in the same directory
import App4 from './pages/App4'; // Assuming App4 is in the same directory
const Home = () => {
  const [mode, setMode] = useState<'app1' | 'app2' |'app3'| null>(null);

  if (mode === 'app1') return <App />;
  if (mode === 'app2') return <App2 />;
  if (mode === 'app3') return <App4 />;

  return (
    <View style={styles.container}>
      {/* <Image src='./assets/eye.png' ></Image> */}
      <Image source={require('./assets/eye.png')} style={{ width: 300, height: 200, marginBottom: 20 }} />
      <Text style={styles.title}>Select App Mode</Text>
      <TouchableOpacity style={styles.button} onPress={() => setMode('app1')}>
        <Text style={styles.buttonText}>Create Dataset</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.button} onPress={() => setMode('app2')}>
        <Text style={styles.buttonText}>Test Application</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.button} onPress={() => setMode('app3')}>
        <Text style={styles.buttonText}>VideoMode</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center' ,backgroundColor: '#f0f8ff'},
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

export default Home;