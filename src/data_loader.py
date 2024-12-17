import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

label_mapping = {'palm': 0, 'point': 1, 'grip': 2, 'like': 3, 'dislike': 4, 'no_gesture':5}

# הפונקציה שמבצעת flattening על ה-coordinates
def flatten_Coordinates(hand_landmarks, target_length):
   
    # flatten את כל הנקודות למערך שטוח
    flat_landmarks = [coordinate for landmark in hand_landmarks for coordinate in landmark]
    flat_landmarks = np.array(flat_landmarks).flatten()

    # הוספת ריפוד אם הווקטור קצר מהאורך הרצוי
    if len(flat_landmarks) < target_length:
        padding = [0.0] * (target_length - len(flat_landmarks))  # ריפוד עם אפסים
        flat_landmarks = np.concatenate((flat_landmarks, padding))

    # חיתוך אם הווקטור ארוך מהאורך הרצוי
    elif len(flat_landmarks) > target_length:
        flat_landmarks = flat_landmarks[:target_length]

    return flat_landmarks


# קריאת הקובץ
def load_data(filename, maxTarget):
    with open(filename, 'r') as f:
        data = json.load(f)

    X = []
    Y = []

    # הדפסת חלק מהנתונים (כדי לוודא שהנתונים נקראו כראוי)
    for key, obj in data.items():  # מעבר על המילון
        hand_landmarks = obj['hand_landmarks']
        label = obj['labels'][0]  # אנחנו מניחים שהתווית היא ב-indx 0
        
        # flatten את ה-hand_landmarks
        flattened_landmarks = flatten_Coordinates(hand_landmarks, maxTarget)
        
        # המרת התווית בעזרת LabelEncoder
        encoded_label = label_mapping[label]
        
        X.append(flattened_landmarks)
        Y.append(encoded_label)

    return np.array(X), np.array(Y)

# טעינת נתונים ממספר קבצים
def load_data_from_files(file_paths):
    """
    טוען נתונים ממספר קבצים ומאחד את כל הדוגמאות ל-X ו-Y.
    """
    X, Y = [], []

    maxTarget =get_max_length(file_paths)

    for file_path in file_paths:
        x_data, y_data = load_data(file_path,maxTarget)
        X.extend(x_data)
        Y.extend(y_data)

    return np.array(X), np.array(Y)

def get_max_length(dataset_files):
    max_length = 0
    
    for file_path in dataset_files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            for key, obj in data.items():
                hand_landmarks = obj['hand_landmarks']
                flat_landmarks = [coordinate for landmark in hand_landmarks for coordinate in landmark]
                max_length = max(max_length, len(flat_landmarks))
    
    return max_length







