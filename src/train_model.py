# train_model.py
import numpy as np
from model_builder import build_model  # הנח שיש לך מודל מוגדר בקובץ אחר  
from data_loader import load_data_from_files  # אם יש לך פונקציה שמביאה את הנתונים

# שלב 1: הגדרת הנתיבים של כל הקבצים
dataset_files = [
    'dataset/like.json',
    'dataset/dislike.json',
    'dataset/palm.json',
    'dataset/grip.json',
    'dataset/point.json'
]

# שלב 1: טוענים את הנתונים
# שלב 2: טעינת כל הנתונים לקובץ אחד
X, Y = load_data_from_files(dataset_files)

# שלב 2: בניית המודל
input_shape = (X.shape[1],)  # צורת הקלט של המודל (כמו מספר הקואורדינטות שלך)
num_classes = len(np.unique(Y))  # מספר המחוות השונות (תוויות)
model = build_model(input_shape, num_classes)

# שלב 3: אימון המודל
model.fit(X, Y, epochs=10, batch_size=32)

# שמירת המודל לאחר האימון
model.save('hand_gesture_model.h5')
