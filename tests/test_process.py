# לא לשכוח למחוק!!!

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# הגדרות
NUM_CLASSES = 6   # מספר הקטגוריות לסיווג
INPUT_SHAPE = (50, 2, 1)  # גודל הקלט: 50 נקודות ציון (פדינג במידת הצורך), עם 2 מאפיינים X ו-Y

# יצירת דאטה מדומה
num_samples = 1000  # מספר הדוגמאות לאימון
features = [np.random.rand(np.random.randint(10, 50), 2) for _ in range(num_samples)]  # מערך עם נקודות ציון משתנות
labels = np.random.randint(0, NUM_CLASSES, size=(num_samples,))  # תוויות אקראיות בין 0 ל-5

# Padding לנקודות כך שכולן יהיו באורך 50
def pad_features(data, max_length=50):
    padded_data = np.zeros((len(data), max_length, 2))  # מאתחל עם אפסים
    for i, sample in enumerate(data):
        padded_data[i, :len(sample), :] = sample  # מציב את הנקודות במיקום הנכון
    return padded_data

X_padded = pad_features(features)
X_padded = X_padded[..., np.newaxis]  # הוספת ערוץ אחד (ל-CNN)
y_categorical = to_categorical(labels, NUM_CLASSES)  # המרת תוויות ל-one-hot

# חלוקה לסט אימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)

# בניית מודל CNN
model = Sequential([
    Conv2D(32, (3, 2), activation='relu', padding='same', input_shape=INPUT_SHAPE),
    MaxPooling2D(pool_size=(2, 1)),
    
    Conv2D(64, (3, 2), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 1)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

# קומפילציה
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# אימון
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# הערכת המודל
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy on test set: {accuracy:.2f}")
