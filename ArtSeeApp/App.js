import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { Camera } from 'expo-camera';
import { auth } from './firebaseConfig';

function App() {
  const [userId, setUserId] = useState(null);
  const [hasCameraPermission, setHasCameraPermission] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.back);
  const [messageOpacity] = useState(new Animated.Value(1));

  useEffect(() => {
    (async () => {
      try {
        const { status } = await Camera.requestCameraPermissionsAsync();
        setHasCameraPermission(status === 'granted');

        setTimeout(() => {
          Animated.timing(messageOpacity, {
            toValue: 0.3,
            duration: 2000,
            useNativeDriver: true,
          }).start();
        }, 5000);
      } catch (err) {
        console.error("Error requesting camera permissions:", err);
      }
    })();

    const unsubscribe = auth.onAuthStateChanged(async user => {
      if (user) {
        setUserId(user.uid);
      } else {
        auth.signInAnonymously().catch(error => {
          console.error('Error signing in anonymously:', error);
        });
      }
    });

    return unsubscribe;
  }, []);

  if (hasCameraPermission === null) {
    return <View />;
  }

  if (hasCameraPermission === false) {
    return <Text>No access to camera</Text>;
  }

  return (
    <View style={styles.container}>
      <Camera style={styles.camera} type={type}>
        <Animated.View style={{...styles.messageContainer, opacity: messageOpacity}}>
          <Text style={styles.messageText}>Point the camera towards a painting</Text>
        </Animated.View>
      </Camera>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  messageContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 50,
  },
  messageText: {
    fontSize: 20,
    color: 'white',
    textShadowColor: 'rgba(0, 0, 0, 0.5)',
    textShadowOffset: { width: -1, height: 1 },
    textShadowRadius: 10,
  },
});

export default App;
