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

def convert_input_to_array(input_data):
    """
    ממירה רשימה של מילונים למערך numpy המתאים לקלט המודל.
    כל מילון מומר למערך [x, y].
    
    :param input_data: רשימה של מילונים עם המפתחות 'x' ו-'y'.
    :return: מערך numpy בגודל (1, time_steps, 2).
    """
    # הפיכת המילונים לרשימה של רשימות [x, y]
    converted_data = [[point['x'], point['y']] for point in input_data[0]]
    
    # המרה למערך numpy והוספת ממד ראשון (batch size)
    return np.array([converted_data], dtype=np.float32)
new_input_features,t=load_data('data/output/result.json',42,True)
# data,new_input = process_image("data\\palm.jpeg")

# #print(flatten_Coordinates(convert_input_to_array(new_input),42).shape)
# #print(convert_input_to_array(new_input)[0])
#    # חיזוי בעזרת המודל
# prediction = model.predict(np.array([[0.4253077 , 0.8778372 ],
#        [0.5786977 , 0.80821514],
#        [0.69370943, 0.6865734 ],
#        [0.76201683, 0.56195796],
#        [0.83656836, 0.4754991 ],
#        [0.5572408 , 0.47157124],
#        [0.5774329 , 0.29879647],
#        [0.588399  , 0.19303623],
#        [0.5949183 , 0.10011345],
#        [0.46201143, 0.4617218 ],
#        [0.4630314 , 0.28272736],
#        [0.46499395, 0.1682826 ],
#        [0.46593177, 0.07301268],
#        [0.37390673, 0.48605275],
#        [0.3609959 , 0.32040113],
#        [0.35886654, 0.21600407],
#        [0.36004448, 0.12867588],
#        [0.28760362, 0.5394994 ],
#        [0.24467117, 0.41248232],
#        [0.22155523, 0.33095038],
#        [0.20601702, 0.25385135],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ],
#        [2.        , 2.        ]])
# )
print(new_input_features)
prediction = model.predict(new_input_features)
# הדפסת התוצאה
print("\nPrediction:")
print(prediction)