# train_model.py
import os
import numpy as np
#import time
import keras as ker
#from model_builder import build_model,prepare_datasets
from data_loader import load_data
#from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
#import seaborn as sns
import matplotlib.pyplot as plt

# Mapping of gesture labels to numerical values
label_mapping = {'palm': 0, 'point': 1, 'grip': 2, 'like': 3, 'dislike': 4, 'no_gesture': 5}
# List of model names to load and use for prediction
models_names = ["hand_gesture_model"]
# List of gesture label names
label_names = [key for key, val in label_mapping.items()]

# Path to the folder containing the files with hand gesture data
folder_path = r"C:\Users\elino\Desktop\לימודים\שנה ג\סמסטר א\בינה מלאכותית ויישומה\RecipeReach\data\output"
load_path = r"data/output"

# Number of gesture categories
num_categories = len(label_mapping)

def predict_gesture(model, json_file_path, name):
    """
    Function to predict the gesture from the hand data using a pre-trained model.
    
    Args:
        model: Pre-trained Keras model for hand gesture recognition.
        json_file_path: Path to the JSON file containing the hand data for prediction.
        name: The name of the file or data (for example, gesture file name).
        
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

# Loop over each model name in the list
for name in models_names:
    # Load the pre-trained model
    model = ker.models.load_model(f'{name}.keras')
    
    # Loop over all files in the folder containing the gesture data
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            # Predict the gesture for the current file
            predict_gesture(model, file_path, file_name)
