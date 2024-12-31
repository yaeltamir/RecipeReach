import cv2
import mediapipe as mp
from   process_image import process_image
import numpy as np
import uuid
import json
#from   train_model import predict_gesture 
import keras as ker
from data_loader import load_data


label_mapping = {'palm': 0, 'point': 1, 'grip': 2, 'like': 3, 'dislike': 4, 'no_gesture': 5}
label_names=[key for key,val in label_mapping.items() ]

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


cap = cv2.VideoCapture(0)  # פתיחת מצלמה

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # עיבוד פריים
    processed_data = process_image(frame)

    # אם זוהתה יד
    if processed_data["hand_landmarks"]:

        # שמירת הנתונים בקובץ JSON זמני
        json_file_path = "data/output/temp_data.json"
        with open(json_file_path, "w") as f:
            json.dump({"temp":processed_data}, f, indent=4)

        model = ker.models.load_model('hand_gesture_model.keras')
        # שליחה לרשת הנוירונים
        predict_gesture(model, json_file_path, "Real-Time Frame")

    # הצגת פריים
    cv2.imshow("Real-Time Hand Gesture Detection", frame)

    # יציאה בלחיצה על מקש 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

