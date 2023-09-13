import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, Image } from 'react-native'; // Import Image component
import axios from 'axios';

function App() {
  const [url, setUrl] = useState('');

  const handleSend = async () => {
    try {
      const response = await axios.post('http://192.168.0.124:8000/store_url', { url });
      console.log(response.data.message);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Enter URL:</Text>
      <TextInput
        style={styles.input}
        value={url}
        onChangeText={(text) => setUrl(text)}
        placeholder="https://example.com"
      />
      <Button title="Send" onPress={handleSend} />

      {/* Add the Image component with the source */}
      <Image
        source={require('./output_image.png')} // Change the source path as needed
        style={styles.image}
      />
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
    width: 200, // Set the width of the image as needed
    height: 200, // Set the height of the image as needed
    marginTop: 20, // Add spacing between the button and the image
  },
});

export default App;
