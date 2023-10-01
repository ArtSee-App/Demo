import tensorflow as tf

# Load the model
model = tf.saved_model.load("best.pb")

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model("best.pb")
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
