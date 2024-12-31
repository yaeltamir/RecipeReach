import keras as ker
import numpy as np
from data_loader import load_data_from_files

DATASET_PATH='dataset'
datasetTrain_files=[
    'dataset/like_train.json',
    'dataset/dislike_train.json',
    'dataset/palm_train.json',
    'dataset/grip_train.json',
    'dataset/point_train.json',
    'dataset/no_gesture_train.json'
    ]

datasetTest_files=[
    'dataset/like_test.json',
    'dataset/dislike_test.json',
    'dataset/palm__test.json',
    'dataset/grip_test.json',
    'dataset/point_test.json',
    'dataset/no_gesture__test.json'
]

datasetValidate_files = [
    'dataset/like.json',
    'dataset/dislike.json',
    'dataset/palm.json',
    'dataset/grip.json',
    'dataset/point.json',
    'dataset/no_gesture.json'
]

def build_model(input_shape, num_classes):
    """
    בונה את המודל לרשת הנוירונים.
    :param input_shape: צורת הקלט (אורך וקטור הנתונים)
    :param num_classes: מספר המחלקות לסיווג
    :return: המודל המוגדר
    """
    model = ker.models.Sequential() # בונה מודל סדרתי כך שהרשת נוירונים תהיה שכבה אחרי שכבה
    model.add(ker.layers.Masking(mask_value=[-2.0,-2.0], input_shape=input_shape))  # מתעלם מערכים שהוספנו לריפוד, במקרה זה הערכים מוגדרים כ-2
   
    model.add(ker.layers.Flatten(input_shape=input_shape))  # הפיכת הקלט לוקטור אחד-----------------------------------------
    #Dense אומר שכל שכבת נוירונים תהיה מחוברת לשכבה קודמת, כאן עושים 256 
    model.add(ker.layers.Dense(256, activation='relu')) 
    model.add( ker.layers.Dropout(0.4)) 
    #ReLU מחזירה את הערך עצמו אם הוא חיובי או 0 אם הוא שלילי 
    model.add(ker.layers.Dense(128, activation='relu'))  
    # Dropout להורדת סיכוי לאוברפיטינג
    # במקרה זה יבחרו 35 אחוז מהנוירונים באופן רנדומלי והם לא יפעלו
    model.add(ker.layers.Dropout(0.35))  
    model.add( ker.layers.Dense(64, activation='relu'))  
    model.add( ker.layers.Dropout(0.3))                                              
    model.add( ker.layers.Dense(32, activation='relu'))
    # שכבת הפלט לסיווג
    # מגדיר שהשכבה האחרונה של הנוירונים תהיה כמספר הסיווגים השונים שיש לי 
    #Softmax ממירה את הפלט לווקטור של הסתברויות שסכומן הוא 1.
    model.add(ker.layers.Dense(num_classes, activation='softmax'))  
    return model

# def train_model():
#     num_categories=len(np.unique(labels))
#     input_shape = (features.shape[1],features.shape[2])  

def prepare_datasets():
    features_train,labels_train=load_data_from_files(datasetTrain_files)
    num_categories=len(np.unique(labels_train))
    labels_train = ker.utils.to_categorical(labels_train, num_classes=num_categories)

    features_test,labels_test=load_data_from_files(datasetTest_files,features_train.shape[1])
    labels_test = ker.utils.to_categorical(labels_test, num_classes=num_categories)

    features_validate,labels_validate=load_data_from_files(datasetValidate_files,features_train.shape[1])
    labels_validate = ker.utils.to_categorical(labels_validate, num_classes=num_categories)

    return {'features':features_train,'labels':labels_train},\
           {'features':features_validate,'labels':labels_validate},\
           {'features':features_test,'labels':labels_test}

    



