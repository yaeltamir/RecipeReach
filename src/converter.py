import tensorflow as tf

# Load the pre-trained hand gesture recognition model
model = tf.keras.models.load_model("hand_gesture_model.keras")

# Convert the model to TensorFlow Lite format for deployment on edge devices
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted TFLite model to a file
with open("hand_gesture_model.tflite", "wb") as f:
    f.write(tflite_model)
