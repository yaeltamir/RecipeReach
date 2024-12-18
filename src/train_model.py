# train_model.py
import numpy as np
import time
import keras as ker
from model_builder import build_model  # הנח שיש לך מודל מוגדר בקובץ אחר  
from data_loader import load_data_from_files  # אם יש לך פונקציה שמביאה את הנתונים


# שמירת הזמן בתחילת הריצה
start_time = time.time()

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
fetures, labels = load_data_from_files(dataset_files)
# המרה ל-one-hot encoding
print("features:(number of sumples, num of attributes) =", fetures.shape) 

num_categories=len(np.unique(labels))
print("num of different categories:",num_categories)

labels = ker.utils.to_categorical(labels, num_classes=num_categories)
print("labels:(number of sumples, num of attributes) =", labels.shape)

# שלב 2: בניית המודל
#מגדיר שהקלט שהמודל יקבל בעתיד יהיה מהצורה ([ מערך של כל נקודות הציון של המפרקים בגודל של מה שהגדרנו בפנים =fetures.shape[1]=כמספר התכונות שיש במערך])
#יש פסיק בהגדרה כדי להגדיר שזה טאפל ממימד 1 אחרת זה סתם משתנה עם סוגריים
input_shape = (fetures.shape[1],)  

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
model.fit(fetures, labels, epochs=10, batch_size=32)
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
