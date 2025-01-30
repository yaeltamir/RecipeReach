# import os
# import json
# import numpy as np
import json
import numpy as np

label_mapping = {'palm': 0, 'point': 1, 'grip': 2, 'like': 3, 'dislike': 4, 'no_gesture':5}

# # הפונקציה שמבצעת flattening על ה-coordinates
# def flatten_Coordinates(hand_landmarks, target_length):
   
#     # flatten את כל הנקודות למערך שטוח
#     flat_landmarks = [coordinate for landmark in hand_landmarks for coordinate in landmark]
#     flat_landmarks = np.array(flat_landmarks).flatten()

#     # הוספת ריפוד אם הווקטור קצר מהאורך הרצוי
#     if len(flat_landmarks) < target_length:
#         padding = [-0.1] * (target_length - len(flat_landmarks))  # ריפוד עם אפסים
#         flat_landmarks = np.concatenate((flat_landmarks, padding))

#     # חיתוך אם הווקטור ארוך מהאורך הרצוי
#     elif len(flat_landmarks) > target_length:
#         flat_landmarks = flat_landmarks[:target_length]

#     return flat_landmarks


# # קריאת הקובץ
# def load_data(filename, maxTarget):
    
#     with open(filename, 'r') as f:
#         data = json.load(f)

#     hand_coordinates = []
#     gesture_label = []

#     # הדפסת חלק מהנתונים (כדי לוודא שהנתונים נקראו כראוי)
#     for key, obj in data.items():  # מעבר על המילון
#         hand_landmarks = obj['hand_landmarks']
#         label = obj['labels'][0]  # אנחנו מניחים שהתווית היא ב-indx 0
        
#         # flatten את ה-hand_landmarks
#         flattened_landmarks = flatten_Coordinates(hand_landmarks, maxTarget)
        
#         # המרת התווית בעזרת LabelEncoder
#         encoded_label = label_mapping[label]
        
#         hand_coordinates.append(flattened_landmarks)
#         gesture_label.append(encoded_label)

#     return np.array(hand_coordinates), np.array(gesture_label)

# טעינת נתונים ממספר קבצים
def load_data_from_files(file_paths,maxTarget=-1):
    """
    טוען נתונים ממספר קבצים ומאחד את כל הדוגמאות ל-X ו-Y.
    """
    X, Y = [], []

    if maxTarget==-1:
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





def flatten_Coordinates(hand_landmarks, target_length):
    """
    מאחד את כל הנקודות כך שגודל הרשימה יהיה אחיד (target_length).
    מוסיף ריפוד (0.0, 0.0) אם אין מספיק נקודות או חותך נקודות עודפות.
    """
    hand_landmarks = np.array(hand_landmarks)  # המרה למערך Numpy
    
  
    # print("fl_f:",hand_landmarks)

    if len(hand_landmarks) < target_length:
        # ריפוד עם נקודות אפס (0,0)
        padding = [[-2.0, -2.0]] * (target_length - len(hand_landmarks))
        #print(padding)
        hand_landmarks = np.vstack((hand_landmarks, padding))
    
    elif len(hand_landmarks) > target_length:
        # חיתוך לנקודות עודפות
        hand_landmarks = hand_landmarks[:target_length]
        
    return hand_landmarks


def load_data(filename, maxTarget=21, istest=False):
    """
    Reads a JSON file and prepares the data for model training.
    
    :param filename: Name of the file
    :param maxTarget: Desired number of hand keypoints
    :param istest: Boolean flag indicating if the data is for testing (default is False)
    :return: NumPy arrays of features (X) and labels (Y)
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    X = []
    Y = []

    for obj in data.values():
        hand_landmarks = obj['hand_landmarks'][0]  # Extracts the first set of hand landmarks

        if len(hand_landmarks) == 0:
            continue
        
        if not istest:
            label = obj['labels']  # Retrieves the label
            # Converts the classification into a numerical value (e.g., "like" → 0)
            Y.append(label_mapping[label[0]])

        # Ensures a uniform size for the number of hand keypoints
        flattened_landmarks = flatten_Coordinates(hand_landmarks, maxTarget)
        X.append(flattened_landmarks)

    return np.array(X), np.array(Y)





