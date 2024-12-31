# train_model.py
import os
import numpy as np
import time
import keras as ker
from model_builder import build_model,prepare_datasets
from data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

label_mapping = {'palm': 0, 'point': 1, 'grip': 2, 'like': 3, 'dislike': 4, 'no_gesture': 5}
models_names=["hand_gesture_model_withTheGivenSplit","hand_gesture_model_fitOnlyByTrainSet"]
label_names=[key for key,val in label_mapping.items() ]

# מסלול לתיקייה
folder_path = r"C:\Users\elino\Desktop\לימודים\שנה ג\סמסטר א\בינה מלאכותית ויישומה\RecipeReach\data\output"
load_path=r"data/output"

num_categories=len(label_mapping)

def predict_gesture(model, json_file_path,name):
    with open("model_input_length.txt", "r") as f:
        model_input_length= int(f.read())
    new_input_features,trash=load_data(json_file_path,model_input_length,True)
    prediction = model.predict(new_input_features)
    predicted_label_index = np.argmax(prediction, axis=1)  # האינדקס של הערך המקסימלי
    confidence_scores = np.max(prediction, axis=1)  # הערך המקסימלי (מידת הביטחון)
    predicted_label = label_names[predicted_label_index[0]]
    print(f"{name} => Prediction:",end="   ")
    print(f"Label: {predicted_label}, Confidence: {confidence_scores[0]:.5%}")


for name in models_names:
    model = ker.models.load_model(f'{name}.keras')
    # לולאה שעוברת על כל הקבצים בתיקייה
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # בדיקה אם זה קובץ (ולא תיקייה)
        if os.path.isfile(file_path):
            predict_gesture(model,file_path,file_name)
