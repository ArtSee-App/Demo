import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, Image } from 'react-native';
import { firestore, storage, ref } from './firebaseConfig'; // Import Firebase from your configuration file
import * as ImagePicker from 'expo-image-picker'; // Import from expo-image-picker
import { uploadBytes } from 'firebase/storage';
import { getStorage, getDownloadURL } from 'firebase/storage'; // Import Firebase Storage functions

function App() {
  const [url, setUrl] = useState('');
  const [imageUri, setImageUri] = useState(null);

  const getPermissionAsync = async () => {
    if (Platform.OS !== 'web') {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        alert('Sorry, we need media library permissions to make this work.');
      }
    }
  };

  useEffect(() => {
    getPermissionAsync(); // Request permission when the component mounts
    const storage = getStorage(); // Get a reference to Firebase Storage

    const storageRef = ref(storage, 'test/output_image'); // Replace 'output_image' with the actual image name
    getDownloadURL(storageRef)
      .then((downloadURL) => {
        // Set the URL in the state
        setImageUri(downloadURL);
      })
      .catch((error) => {
        console.error('Error fetching image from Firebase Storage:', error);
      });

  }, []);


  const handleCameraCapture = async () => {
    let result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      // Upload the captured image to Firebase Storage
      const response = await fetch(result.assets[0].uri);
      const blob = await response.blob();
      const reference = ref(
        storage,
        'test/input_image'
      );
      uploadBytes(reference, blob);

    }
  };

  
  return (
    <View style={styles.container}>
      {imageUri && <Image source={{ uri: imageUri }} style={styles.image} />}
      <Button title="Take a Picture" onPress={handleCameraCapture} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  label: {
    fontSize: 18,
    marginBottom: 10,
  },
  input: {
    width: '100%',
    height: 40,
    borderWidth: 1,
    borderColor: 'gray',
    paddingHorizontal: 10,
    marginBottom: 20,
  },
  image: {
    width: 200,
    height: 200,
    marginBottom: 20,
  },
});

export default App;
