import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, Image } from 'react-native';
import { auth, storage, ref } from './firebaseConfig'; // Import Firebase from your configuration file
import * as ImagePicker from 'expo-image-picker'; // Import from expo-image-picker
import { uploadBytes } from 'firebase/storage';
import { getStorage, getDownloadURL, list, listAll } from 'firebase/storage';
import { getFirestore, doc, getDoc, setDoc } from 'firebase/firestore';
import axios from 'axios';  // <-- Import axios

function App() {
  const [imageUri, setImageUri] = useState(null);
  const [refreshCount, setRefreshCount] = useState(0);  // Step 1
  const [userId, setUserId] = useState(null);

  const getPermissionAsync = async () => {
    if (Platform.OS !== 'web') {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        alert('Sorry, we need media library permissions to make this work.');
      }
    }
  };
  const sendUserIdToAPI = async (userId) => {
    try {
      const response = await axios.post('http://192.168.0.124:5000/user', {
        uid: userId  // send user's UID in request body
      });
      console.log('Data sent successfully:', response.data);
    } catch (error) {
      console.error('Error sending UID to API:', error);
    }
  };
  

  

    
  useEffect(() => {
    getPermissionAsync(); // Request permission when the component mounts

    // Set an interval to refresh the component logic every second
    const interval = setInterval(() => {
      setRefreshCount((prevCount) => prevCount + 1);  // Step 2
    }, 1000);

    // Clear the interval when the component unmounts
    return () => {
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(async user => {
      if (user) {
        setUserId(user.uid); // Set the user's UID to the state
        
        const db = getFirestore();
        const userDocRef = doc(db, 'indeces', user.uid); // Using the user's UID as the document ID
        const userDocSnapshot = await getDoc(userDocRef);
  
        // If the user doesn't already have a document, create one
        if (!userDocSnapshot.exists()) {
          await setDoc(userDocRef, { index: 1 });
        }
  
  
        // Move the logic from the second useEffect here, inside the condition where user is defined
        // Fetch the value from Firestore using the user's UID
        const docRef = doc(db, 'indeces', user.uid);
  
        getDoc(docRef).then((documentSnapshot) => {
          if (documentSnapshot.exists()) {
            const fieldValue = documentSnapshot.data().index.toString();
            // Now, use this fieldValue to list files with the prefix in Firebase Storage
            const storage = getStorage();
            const storageRef = ref(storage, 'artworks'); // reference to 'artworks' folder
            
            // List all files with the prefix
            listAll(storageRef).then((result) => {
              const matchingFiles = result.items.filter(item => item.name.startsWith(fieldValue));
  
              if (matchingFiles.length > 0) {
                // Use the first file that matches the prefix
                getDownloadURL(matchingFiles[0])
                  .then((downloadURL) => {
                    setImageUri(downloadURL);
                  })
                  .catch((error) => {
                    console.error('Error fetching image URL:', error);
                  });
              } else {
                console.error('No files found with the given prefix:', fieldValue);
              }
            }).catch((error) => {
              console.error('Error listing files from Firebase Storage:', error);
            });
          }
        }).catch((error) => {
          console.error('Error fetching index value from Firestore:', error);
        });
  
      } else {
        auth.signInAnonymously().catch(error => {
          console.error('Error signing in anonymously:', error);
        });
      }
    });
  
    // Clean up function
    return unsubscribe;
  }, [refreshCount]); // Only depend on refreshCount because userId is being set within this useEffect
  

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
      
      // Once the image is uploaded to Firebase Storage, call the model API
      await uploadBytes(reference, blob).then(() => {
        callRunModelAPI();
      });
    }
};

  
  const callRunModelAPI = async () => {
    try {
      sendUserIdToAPI(userId); // Send UID to API
      const response = await axios.post('http://192.168.0.124:5000/run-model');
      // handle the response from the Flask API if needed
      console.log(response.data);
    } catch (error) {
      console.error('Error calling /run-model:', error);
    }
  };
return (
  <View style={styles.container}>
    {userId && <Text>User ID: {userId}</Text>} 
    {imageUri && <Image source={{ uri: imageUri }} style={styles.image} />}
    <Button title="Take a Picture and Send" onPress={handleCameraCapture} />
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
