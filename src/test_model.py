from tensorflow import keras
from model_builder import build_model,prepare_datasets

model = keras.models.load_model("hand_gesture_model.keras")
# train,validate,test=prepare_datasets(6)
# loss, accuracy = model.evaluate(test['features'], test['labels'], verbose=1)
# print(f"Accuracy: {accuracy * 100:.2f}%")
model.summary()


 