import tensorflow as tf

model = tf.keras.models.load_model("hand_gesture_model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("hand_gesture_model.tflite", "wb") as f:
    f.write(tflite_model)
