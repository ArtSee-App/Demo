// Import the functions you need from the SDKs you need
import firebase from "firebase/compat" // TODO: Add SDKs for Firebase products that you want to use
import { getStorage, ref  } from "firebase/storage";

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBBZrptsnMDFh25B0LvcBmp-n3CF0odUNA",
  authDomain: "artsee-42c23.firebaseapp.com",
  projectId: "artsee-42c23",
  storageBucket: "artsee-42c23.appspot.com",
  messagingSenderId: "587875980821",
  appId: "1:587875980821:web:51360fb6c025ffacd7e01c",
  measurementId: "G-CCQQ4RW6BG"
};

// Initialize Firebase
let app;
if (firebase.apps.length === 0) {
  app = firebase.initializeApp(firebaseConfig);
} else {
  app = firebase.app()
}

const auth = firebase.auth()
const firestore = firebase.firestore(); // Add Firestore instance
const storage = getStorage(app);

export { auth, firestore, storage, ref }; // Export both auth and firestore