from tensorflow import keras
from model_builder import prepare_datasets

# Load the pre-trained hand gesture recognition model
model = keras.models.load_model("hand_gesture_model.keras")

# Prepare datasets with 6 gesture categories (train, validate, and test)
train, validate, test = prepare_datasets(6)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test['features'], test['labels'], verbose=1)

# Print the accuracy percentage
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display the model's architecture summary
model.summary()
