import keras as ker


def build_model(input_shape, num_classes):
    """
    בונה את המודל לרשת הנוירונים.
    :param input_shape: צורת הקלט (אורך וקטור הנתונים)
    :param num_classes: מספר המחלקות לסיווג
    :return: המודל המוגדר
    """
    model = ker.models.Sequential() # בונה מודל סדרתי כך שהרשת נוירונים תהיה שכבה אחרי שכבה
    model.add(ker.layers.Masking(mask_value=2.0, input_shape=input_shape))  # מתעלם מערכים שהוספנו לריפוד, במקרה זה הערכים מוגדרים כ-0.1
    model.add(ker.layers.Flatten(input_shape=input_shape))  # הפיכת הקלט לוקטור אחד-----------------------------------------
    #Dense אומר שכל שכבת נוירונים תהיה מחוברת לשכבה קודמת, כאן עושים 128 
    #ReLU מחזירה את הערך עצמו אם הוא חיובי או 0 אם הוא שלילי 
    model.add(ker.layers.Dense(128, activation='relu'))  

# Dropout להורדת סיכוי לאוברפיטינג
# במקרה זה יבחרו 30 אחוז מהנוירונים באופן רנדומלי והם לא יפעלו
    model.add(ker.layers.Dropout(0.3))  


# שכבת הפלט לסיווג
# מגדיר שהשכבה האחרונה של הנוירונים תהיה כמספר הסיווגים השונים שיש לי 
#Softmax ממירה את הפלט לווקטור של הסתברויות שסכומן הוא 1.
    model.add(ker.layers.Dense(num_classes, activation='softmax'))  
    return model



'''
import keras as ker

def build_model(input_shape, num_classes):
    """
    בונה את המודל לסיווג מחוות יד.
    :param input_shape: צורת הקלט (למשל 42 עבור 21 נקודות × 2)
    :param num_classes: מספר הקטגוריות לפלט
    """
    model = ker.models.Sequential()
    model.add(ker.layers.Flatten(input_shape=input_shape))  # הפיכת הקלט לוקטור אחד
    model.add(ker.layers.Dense(128, activation='relu'))  # שכבת נוירונים עם ReLU
    model.add(ker.layers.Dropout(0.3))
    model.add(ker.layers.Dense(num_classes, activation='softmax'))  # שכבת פלט

    return model

'''
