# train_model.py
import numpy as np
import time
import keras as ker
from model_builder import build_model  # הנח שיש לך מודל מוגדר בקובץ אחר  
from data_loader import load_data_from_files,flatten_Coordinates ,load_data # אם יש לך פונקציה שמביאה את הנתונים
from process_image import process_image


# שמירת הזמן בתחילת הריצה
start_time = time.time()

# שלב 1: הגדרת הנתיבים של כל הקבצים
dataset_files = [
    'dataset/like.json',
    'dataset/dislike.json',
    'dataset/palm.json',
    'dataset/grip.json',
    'dataset/point.json'
    ,'dataset/no_gesture.json'
]

# שלב 1: טוענים את הנתונים
# שלב 2: טעינת כל הנתונים לקובץ אחד
features, labels = load_data_from_files(dataset_files)
# המרה ל-one-hot encoding
print("features:(number of sumples, num of attributes) =", features.shape) 

num_categories=len(np.unique(labels))
print("num of different categories:",num_categories)

labels = ker.utils.to_categorical(labels, num_classes=num_categories)
print("labels:(number of sumples, num of attributes) =", labels.shape)

# שלב 2: בניית המודל
#מגדיר שהקלט שהמודל יקבל בעתיד יהיה מהצורה ([ מערך של כל נקודות הציון של המפרקים בגודל של מה שהגדרנו בפנים =fetures.shape[1]=כמספר התכונות שיש במערך])
#יש פסיק בהגדרה כדי להגדיר שזה טאפל ממימד 1 אחרת זה סתם משתנה עם סוגריים
input_shape = (features.shape[1],features.shape[2])  

start_time_build = time.time()
model = build_model(input_shape, num_categories)

# שמירת הזמן בסוף הריצה
end_time_build = time.time()

'''
optimizer='adam': אלגוריתם אופטימיזציה מתאם, משלב את היתרונות של Momentum ו-RMSprop, ומעדכן את המשקלים באופן אוטומטי במהלך האימון.

loss='categorical_crossentropy': פונקציית אובדן שמתאימה לבעיות סיווג מרובות קטגוריות, ומודדת את השגיאה בין התוצאות החזויות לאמיתיות.

metrics=['accuracy']: מדד לבחינת ביצועי המודל, מחשב את אחוז ההתאמה בין התוצאה החזויה לאמיתית.
'''
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary() #מציג את הארכיטקטורה של המודל, כולל מספר השכבות, סוג השכבות, והפרמטרים שלהן.

start_time_train = time.time()
# שלב 3: אימון המודל
# epochs=10 – מספר הפעמים שהמודל עובר על כל הדאטה.
# batch_size=32 – מספר הדוגמאות בכל אצווה.
model.fit(features, labels, epochs=10, batch_size=32)
end_time_train = time.time()

# שמירת המודל לאחר האימון
model.save('hand_gesture_model.keras')

# שמירת הזמן בסוף הריצה
end_time = time.time()

# חישוב הזמן שחלף
elapsed_time = end_time - start_time

# הדפסת התוצאה
print(f"build model running time : {end_time_build-start_time_build:.4f} sec")
print(f"train model running time : {end_time_train-start_time_train:.4f} sec")
print(f"overall running time : {elapsed_time:.4f} sec")

new_input_features,t=load_data('data/output/result_like.json',42,True)

# טעינת המודל המאומן
model = ker.models.load_model('hand_gesture_model.keras')

# print(new_input_features)
# prediction = model.predict(new_input_features)
# # הדפסת התוצאה
# print("\nPrediction:")
# print(prediction)

label_mapping = {'palm': 0, 'point': 1, 'grip': 2, 'like': 3, 'dislike': 4, 'no_gesture': 5}
# הפוך את המילון כדי למפות מאינדקס לתווית
index_to_label = {v: k for k, v in label_mapping.items()}

# נתוני קלט חדשים
print(new_input_features)
prediction = model.predict(new_input_features)
# מציאת האינדקס של הערך המקסימלי וההסתברות
predicted_label_index = np.argmax(prediction, axis=1)  # האינדקס של הערך המקסימלי
confidence_scores = np.max(prediction, axis=1)  # הערך המקסימלי (מידת הביטחון)

# המרת האינדקס לתוויות
predicted_labels = [index_to_label[idx] for idx in predicted_label_index]

# הדפסת התוצאות
print("\nPrediction:")
for label, confidence in zip(predicted_labels, confidence_scores):
    print(f"Label: {label}, Confidence: {confidence:.2%}")

