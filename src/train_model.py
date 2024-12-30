# train_model.py
import numpy as np
import time
import keras as ker
from model_builder import build_model  # הנח שיש לך מודל מוגדר בקובץ אחר  
from data_loader import load_data_from_files,flatten_Coordinates ,load_data # אם יש לך פונקציה שמביאה את הנתונים
from process_image import process_image


# # # שמירת הזמן בתחילת הריצה
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
print(features[0])
# המרה ל-one-hot encoding
print("features:(number of sumples, num of attributes) =", features.shape) 

categories=np.unique(labels)
num_categories=len(categories)

print("num of different categories:",num_categories)

labels = ker.utils.to_categorical(labels, num_classes=num_categories)
print("labels:(number of sumples, num of attributes) =", labels.shape)

# שלב 2: בניית המודל
#מגדיר שהקלט שהמודל יקבל בעתיד יהיה מהצורה ([ מערך של כל נקודות הציון של המפרקים בגודל של מה שהגדרנו בפנים =fetures.shape[1]=כמספר התכונות שיש במערך])

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
epochs=10 #– מספר הפעמים שהמודל עובר על כל הדאטה.
batch_size=32 #– מספר הדוגמאות בכל אצווה.
#model.fit(features, labels, epochs=10, batch_size=32)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np



# חילוק הנתונים לנתוני אימון ובדיקה
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.25, random_state=42
)

# # מציאת המשקלים
# if len(labels_train.shape) > 1 and labels_train.shape[1] > 1:
#     labels_train = np.argmax(labels_train, axis=1)

# # חישוב המשקלים
# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=categories,
#     y=np.argmax(labels_train, axis=1)
# )
# class_weight_dict = dict(enumerate(class_weights))

# # אימון המודל
#model.fit(features_train, labels_train, epochs=10, batch_size=32, class_weight=class_weight_dict,validation_data=(features_test, labels_test))


# אימון המודל על נתוני האימון
model.fit(features_train, labels_train, epochs=10, batch_size=32, validation_data=(features_test, labels_test))
end_time_train = time.time()

# # # שמירת המודל לאחר האימון
model.save('hand_gesture_model.keras')

# # # שמירת הזמן בסוף הריצה
end_time = time.time()

# # # חישוב הזמן שחלף
elapsed_time = end_time - start_time

# # # הדפסת התוצאה
print(f"build model running time : {end_time_build-start_time_build:.4f} sec")
print(f"train model running time : {end_time_train-start_time_train:.4f} sec")
print(f"overall running time : {elapsed_time:.4f} sec")

#בדיקת הרשת
label_mapping = {'palm': 0, 'point': 1, 'grip': 2, 'like': 3, 'dislike': 4, 'no_gesture': 5}
# הפוך את המילון כדי למפות מאינדקס לתווית
index_to_label = {v: k for k, v in label_mapping.items()}
# טעינת המודל המאומן
model = ker.models.load_model('hand_gesture_model.keras')



import os

# מסלול לתיקייה
folder_path = r"C:\Users\elino\Desktop\לימודים\שנה ג\סמסטר א\בינה מלאכותית ויישומה\RecipeReach\data\output"
load_path=r"data/output"

# לולאה שעוברת על כל הקבצים בתיקייה
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    
    # בדיקה אם זה קובץ (ולא תיקייה)
    if os.path.isfile(file_path):
        new_input_features,t=load_data(file_path,42,True)
        #print(new_input_features)
# prediction = model.predict(new_input_features)
# # הדפסת התוצאה
# print("\nPrediction:")
# print(prediction)
# נתוני קלט חדשים
#print(new_input_features)
        prediction = model.predict(new_input_features)
# מציאת האינדקס של הערך המקסימלי וההסתברות
        predicted_label_index = np.argmax(prediction, axis=1)  # האינדקס של הערך המקסימלי
        confidence_scores = np.max(prediction, axis=1)  # הערך המקסימלי (מידת הביטחון)
        # המרת האינדקס לתוויות
        predicted_labels = [index_to_label[idx] for idx in predicted_label_index]

        print(file_name)
        
        # הדפסת התוצאות
        print(f"{file_name} => Prediction:",end="   ")
        for label, confidence in zip(predicted_labels, confidence_scores):
            print(f"Label: {label}, Confidence: {confidence:.5%}")

