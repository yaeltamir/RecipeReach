import cv2
import mediapipe as mp
from   process_image import process_image
import numpy as np
import uuid
import json
import keras as ker
from data_loader import load_data

# Mapping of gesture labels to numerical values
label_mapping = {'palm': 0, 'point': 1, 'grip': 2, 'like': 3, 'dislike': 4, 'no_gesture': 5}
# List of gesture label names
label_names = [key for key, val in label_mapping.items()]

def predict_gesture(model, json_file_path, name):
    """
    Function to predict the gesture from the hand data using a pre-trained model.
    
    Args:
        model: Pre-trained Keras model for hand gesture recognition.
        json_file_path: Path to the temporary JSON file containing processed hand data.
        name: The name of the input (for example, real-time frame).
        
    Prints:
        The predicted gesture label and the confidence score.
    """
    # Read the model input length from a file
    with open("model_input_length.txt", "r") as f:
        model_input_length = int(f.read())
    
    # Load the processed hand data and discard the second output
    new_input_features, trash = load_data(json_file_path, model_input_length, True)
    
    # Predict the gesture using the model
    prediction = model.predict(new_input_features)
    
    # Get the index of the maximum value in the prediction
    predicted_label_index = np.argmax(prediction, axis=1)
    
    # Get the confidence score of the prediction
    confidence_scores = np.max(prediction, axis=1)
    
    # Get the predicted gesture label
    predicted_label = label_names[predicted_label_index[0]]
    
    # Print the prediction result
    print(f"{name} => Prediction:", end="   ")
    print(f"Label: {predicted_label}, Confidence: {confidence_scores[0]:.5%}")

# Open the webcam for real-time gesture detection
cap = cv2.VideoCapture(0)

while True:
    # Capture each frame from the webcam
    ret, frame = cap.read()
    
    # If no frame is captured, break the loop
    if not ret:
        break

    # Process the captured frame to extract hand landmarks
    processed_data = process_image(frame)

    # If hand landmarks are detected
    if processed_data["hand_landmarks"]:
        # Save the processed data to a temporary JSON file
        json_file_path = "data/output/temp_data.json"
        with open(json_file_path, "w") as f:
            json.dump({"temp": processed_data}, f, indent=4)

        # Load the pre-trained hand gesture model
        model = ker.models.load_model('hand_gesture_model.keras')
        
        # Send the processed data to the model for gesture prediction
        predict_gesture(model, json_file_path, "Real-Time Frame")

    # Display the processed frame with gesture detection
    cv2.imshow("Real-Time Hand Gesture Detection", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
