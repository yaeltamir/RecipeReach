# הפונקציה לעיבוד תמונות
import cv2
import mediapipe as mp
import uuid

def process_image(image_path):
    # מיזוג הכלים של MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # קריאת התמונה
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # הפעלת זיהוי הידיים
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    output = {
        "bboxes": [],
        "united_bbox": None,
        "user_id": str(uuid.uuid4()),  # מזהה משתמש ייחודי לדוגמה
        "hand_landmarks": [],
        "meta": {
            "age": None,  # ניתוח גיל ידרוש כלים נוספים
            "gender": None,  # ניתוח מגדר דורש מודלים אחרים
            "race": None  # ניתוח גזע מחייב שימוש במאגר מתאים
        }
    }

    # עיבוד תוצאות
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # חישוב תיבת הגבול
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * width
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * height
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * width
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * height

            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            output["bboxes"].append(bbox)

            # נקודות ציון
            landmarks = [[lm.x,lm.y] for lm in hand_landmarks.landmark]
            output["hand_landmarks"].append(landmarks)

        # חישוב תיבת גבול מאוחדת אם יש שתי ידיים
        if len(output["bboxes"]) == 2:
            united_x_min = min(output["bboxes"][0][0], output["bboxes"][1][0])
            united_y_min = min(output["bboxes"][0][1], output["bboxes"][1][1])
            united_x_max = max(output["bboxes"][0][0] + output["bboxes"][0][2],
                               output["bboxes"][1][0] + output["bboxes"][1][2])
            united_y_max = max(output["bboxes"][0][1] + output["bboxes"][0][3],
                               output["bboxes"][1][1] + output["bboxes"][1][3])

            output["united_bbox"] = [int(united_x_min), int(united_y_min),
                                     int(united_x_max - united_x_min), int(united_y_max - united_y_min)]

    hands.close()
    return output

