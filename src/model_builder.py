import keras as ker
import numpy as np
from data_loader import load_data_from_files
from sklearn.utils.class_weight import compute_class_weight


datasetTrain_files=[
    'dataset/train/like_train.json',
    'dataset/train/dislike_train.json',
    'dataset/train/palm_train.json',
    'dataset/train/grip_train.json',
    'dataset/train/point_train.json',
    'dataset/train/no_gesture_train.json'
    ]

datasetTest_files=[
    'dataset/test/like_test.json',
    'dataset/test/dislike_test.json',
    'dataset/test/palm__test.json',
    'dataset/test/grip_test.json',
    'dataset/test/point_test.json',
    'dataset/test/no_gesture__test.json'
]

datasetValidate_files = [
    'dataset/like.json',
    'dataset/dislike.json',
    'dataset/palm.json',
    'dataset/grip.json',
    'dataset/point.json',
    'dataset/no_gesture.json'
]



def define_model(input_shape, num_classes):
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
  

def prepare_datasets(num_categories):
    #train section
    features_train,labels_train=load_data_from_files(datasetTrain_files)
    print("features train:(number of sumples, num of attributes,number of inner attributes) =", features_train.shape) 
    # num_categories=len(np.unique(labels_train))
    # print("num of different labels:",num_categories)
    labels_train = ker.utils.to_categorical(labels_train, num_classes=num_categories)
    print("labels train:(number of sumples, num of attributes) =", labels_train.shape)

    #test section
    features_test,labels_test=load_data_from_files(datasetTest_files,features_train.shape[1])
    print("features test:(number of sumples, num of attributes,number of inner attributes) =", features_test.shape) 
    labels_test = ker.utils.to_categorical(labels_test, num_classes=num_categories)
    print("labels test:(number of sumples, num of attributes) =", labels_test.shape)

    #validate section
    features_validate,labels_validate=load_data_from_files(datasetValidate_files,features_train.shape[1])
    print("features validate:(number of sumples, num of attributes,number of inner attributes) =", features_validate.shape) 
    labels_validate = ker.utils.to_categorical(labels_validate, num_classes=num_categories)
    print("labels validate:(number of sumples, num of attributes) =", labels_validate.shape)

    return {'features':features_train,'labels':labels_train},\
           {'features':features_validate,'labels':labels_validate},\
           {'features':features_test,'labels':labels_test}


def build_model(model_name,num_categories,train_set,test_set,num_epochs=10,size_batch=32):
    input_shape=(train_set['features'].shape[1],train_set['features'].shape[2])
    model = define_model(input_shape, num_categories)

    '''
    optimizer='adam': אלגוריתם אופטימיזציה מתאם, משלב את היתרונות של Momentum ו-RMSprop, ומעדכן את המשקלים באופן אוטומטי במהלך האימון.

    loss='categorical_crossentropy': פונקציית אובדן שמתאימה לבעיות סיווג מרובות קטגוריות, ומודדת את השגיאה בין התוצאות החזויות לאמיתיות.

    metrics=['accuracy']: מדד לבחינת ביצועי המודל, מחשב את אחוז ההתאמה בין התוצאה החזויה לאמיתית.
    '''
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # חישוב המשקלים
    class_weights = compute_class_weight(
    class_weight='balanced',  # איזון אוטומטי לפי כמות הדוגמאות
    classes=np.unique(np.argmax(train_set['labels'], axis=1)),  # מחלקות ייחודיות
    y=np.argmax(train_set['labels'], axis=1)  # תוויות המחלקות
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("Class Weights:", class_weight_dict)

    model.fit(train_set['features'], train_set['labels'], epochs=num_epochs, batch_size=size_batch, validation_data=(test_set['features'], test_set['labels']),class_weight=class_weight_dict)

    model.save(f'{model_name}.keras')



    

    



