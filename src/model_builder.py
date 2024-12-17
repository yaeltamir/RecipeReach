import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Masking

def build_model(input_shape, num_classes):
    """
    בונה את המודל לרשת הנוירונים.
    :param input_shape: צורת הקלט (אורך וקטור הנתונים)
    :param num_classes: מספר המחלקות לסיווג
    :return: המודל המוגדר
    """
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))  # מסנן ערכי ריפוד (0.0)
    model.add(Dense(128, activation='relu'))  # שכבת fully-connected עם 128 נוירונים
    model.add(Dropout(0.5))  # Dropout להורדת סיכוי לאוברפיטינג
    model.add(Dense(num_classes, activation='softmax'))  # שכבת הפלט לסיווג
    return model



