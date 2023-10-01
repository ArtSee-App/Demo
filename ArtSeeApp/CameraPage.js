import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Camera } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as jpeg from 'jpeg-js';
import * as ImageManipulator from 'expo-image-manipulator';
import { tensor3d } from '@tensorflow/tfjs';
import { TensorCamera } from '@tensorflow/tfjs-react-native';

export default function App() {
  const [hasPermission, setHasPermission] = useState(null);
  const [model, setModel] = useState(null);
  const [labelsList, setLabelsList] = useState([]);
  const [detectedItem, setDetectedItem] = useState('');
  const cameraRef = useRef(null);
  
  const fetchLabels = async () => {
    const response = await fetch('https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt');
    const labels = await response.text();
    return labels.split('\n').slice(1); // Remove the first line which is "background"
  };
  
  useEffect(() => {
    (async () => {
      // Request camera permissions
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');

      // Load TensorFlow.js
      await tf.ready();

      // Fetch labels
      const labels = await fetchLabels();
      setLabelsList(labels);

      // Load the pre-trained MobileNet model
      const mobilenetModel = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
      setModel(mobilenetModel);
    })();
  }, []);

  const handleImageCapture = async () => {
    // 1. Capture the image from the camera
    const image = await cameraRef.current.takePictureAsync();

    // 2. Preprocess the image: Resize & Convert to JPEG
    const resizedImage = await ImageManipulator.manipulateAsync(
      image.uri,
      [{ resize: { width: 224, height: 224 } }],
      { format: ImageManipulator.SaveFormat.JPEG }
    );

    // Fetch image data
    const rawImageData = new Uint8Array(await (await fetch(resizedImage.uri)).arrayBuffer());
    const { width, height, data } = jpeg.decode(rawImageData, { useTArray: true });

    // Convert the image data to a tensor
    const imageTensor = tf.tensor4d(data, [1, height, width, 4]);

    // Strip the alpha channel to get an RGB tensor
    const imageTensorRGB = tf.slice(imageTensor, [0, 0, 0, 0], [-1, -1, -1, 3]);

    // Pass the tensor to the model for prediction
    const prediction = model.predict(imageTensorRGB);
    console.log(prediction);
    // Find the class index with the maximum probability
    const predictedClassIdxTensor = tf.argMax(prediction, 1);
    const predictedClassIdx = predictedClassIdxTensor.dataSync()[0];
        
    // Map this to the class name
    const predictedClassName = labelsList[predictedClassIdx];
    
    setDetectedItem(predictedClassName);

    // Cleanup tensors
    prediction.dispose();
    imageTensorRGB.dispose();
    imageTensor.dispose();
  };

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  return (
    <View style={styles.container}>
      <Camera style={styles.camera} ref={cameraRef}>
        <Text style={styles.detectedText}>{detectedItem}</Text>
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={styles.button}
            onPress={handleImageCapture}
          >
            <Text style={styles.text}>Detect</Text>
          </TouchableOpacity>
        </View>
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
    justifyContent: 'flex-end',
  },
  detectedText: {
    fontSize: 20,
    marginTop: 10,
    color: 'white',
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 10,
    borderRadius: 5
  },
  buttonContainer: {
    flex: 0.1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
    margin: 20,
  },
  button: {
    flex: 0.5,
    alignSelf: 'flex-end',
    alignItems: 'center',
  },
  text: {
    fontSize: 18,
    marginBottom: 10,
    color: 'white',
  },
});
