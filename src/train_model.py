# train_model.py
import numpy as np
import time
import keras as ker
from model_builder import build_model
from data_loader import load_data_from_files,load_data

# # # שמירת הזמן בתחילת הריצה
#start_time = time.time()

# שלב 1: הגדרת הנתיבים של כל הקבצים
dataset_files = [ 
    # 'dataset/like.json',
    # 'dataset/dislike.json',
    # 'dataset/palm.json',
    # 'dataset/grip.json',
    # 'dataset/point.json',
    # 'dataset/no_gesture.json',
        'dataset/train/like_train.json',
    'dataset/train/dislike_train.json',
    'dataset/train/palm_train.json',
    'dataset/train/grip_train.json',
    'dataset/train/point_train.json',
    'dataset/train/no_gesture_train.json'
]

features, labels = load_data_from_files(dataset_files)
numerical_labels=labels
print("features:(number of sumples, num of attributes) =", features.shape) 

num_labels=len(np.unique(labels))
print("num of different labels:",num_labels)

labels = ker.utils.to_categorical(labels, num_classes=num_labels)
print("labels:(number of sumples, num of attributes) =", labels.shape)

input_shape = (features.shape[1],features.shape[2]) 
#start_time_build = time.time()
model = build_model(input_shape, num_labels)
# שמירת הזמן בסוף הריצה
#end_time_build = time.time()

'''
optimizer='adam': אלגוריתם אופטימיזציה מתאם, משלב את היתרונות של Momentum ו-RMSprop, ומעדכן את המשקלים באופן אוטומטי במהלך האימון.

loss='categorical_crossentropy': פונקציית אובדן שמתאימה לבעיות סיווג מרובות קטגוריות, ומודדת את השגיאה בין התוצאות החזויות לאמיתיות.

metrics=['accuracy']: מדד לבחינת ביצועי המודל, מחשב את אחוז ההתאמה בין התוצאה החזויה לאמיתית.
'''
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary() 

#start_time_train = time.time()
# שלב 3: אימון המודל
# epochs=10 #– מספר הפעמים שהמודל עובר על כל הדאטה.
# batch_size=32 #– מספר הדוגמאות בכל אצווה.
#model.fit(features, labels, epochs=10, batch_size=32)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

datasetTrain_files=[
    'dataset/like_train.json',
    'dataset/dislike_train.json',
    'dataset/palm_train.json',
    'dataset/grip_train.json',
    'dataset/point_train.json',
    'dataset/no_gesture_train.json'
]
# features_train,labels_train=load_data_from_files(datasetTrain_files)
# labels_train = ker.utils.to_categorical(labels_train, num_classes=num_labels)

datasetTest_files=[
    'dataset/like_test.json',
    'dataset/dislike_test.json',
    'dataset/palm__test.json',
    'dataset/grip_test.json',
    'dataset/point_test.json',
    'dataset/no_gesture__test.json'
]

# features_test,labels_test=load_data_from_files(datasetTest_files)
# labels_test = ker.utils.to_categorical(labels_test, num_classes=num_labels)

# # אימון המודל
#model.fit(features_train, labels_train, epochs=10, batch_size=32, class_weight=class_weight_dict,validation_data=(features_test, labels_test))

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.25, random_state=42
)
# חישוב המשקלים
class_weights = compute_class_weight(
    class_weight='balanced',  # איזון אוטומטי לפי כמות הדוגמאות
    classes=np.unique(np.argmax(labels_train, axis=1)),  # מחלקות ייחודיות
    y=np.argmax(labels_train, axis=1)  # תוויות המחלקות
)

class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)

# אימון המודל על נתוני האימון
model.fit(features_train, labels_train, epochs=10, batch_size=32, validation_data=(features_test, labels_test),class_weight=class_weight_dict)
#end_time_train = time.time()

# # # שמירת המודל לאחר האימון
model.save('hand_gesture_model.keras')

# # # # שמירת הזמן בסוף הריצה
# end_time = time.time()

# # # # חישוב הזמן שחלף
# elapsed_time = end_time - start_time

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# חיזוי על נתוני הבדיקה
predictions = model.predict(features_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(labels_test, axis=1)

# הפקת מטריצת בלבול
cm = confusion_matrix(true_labels, predicted_labels)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=numerical_labels, yticklabels=numerical_labels)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()

# דוח ביצועים
#print(classification_report(true_labels, predicted_labels, target_names=labels))


# # # # הדפסת התוצאה
# print(f"build model running time : {end_time_build-start_time_build:.4f} sec")
# print(f"train model running time : {end_time_train-start_time_train:.4f} sec")
# print(f"overall running time : {elapsed_time:.4f} sec")

#בדיקת הרשת
label_mapping = {'palm': 0, 'point': 1, 'grip': 2, 'like': 3, 'dislike': 4, 'no_gesture': 5}
# הפוך את המילון כדי למפות מאינדקס לתווית
index_to_label = {label_name: label_key for label_key, label_name in label_mapping.items()}
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

# print("validation from given dataset:")
# counters=[0,0,0,0]*num_labels #[how many there are,how many right, sum confidence of everyone,sum confidence of who is right]

# for feature,label in features,labels:
#     #cur_label_index=counters[label]
#     counters[label][0]+=1
#     prediction = model.predict(feature)
#     predicted_label_index = np.argmax(prediction, axis=1)  # האינדקס של הערך המקסימלי
#     confidence_scores = np.max(prediction, axis=1)  # הערך המקסימלי (מידת הביטחון)
#     counters[label][2]+=confidence_scores
#     if label==predicted_label_index:
#         counters[label][1]+=1
#         counters[label][3]+=confidence_scores

# for label_key,label_name in label_mapping:
#     print(f"{label_name} => accouracy percentage: {counters[label_key][1]*100/counters[label_key][0]:.3}% , right avg confidence= {counters[label_key][3]*100/counters[label_key][1]:.3}% , overall avg confidence: {counters[label_key][2]*100/counters[label_key][0]:.3}%")
    
    

