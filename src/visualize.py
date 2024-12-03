import cv2

def visualize_bboxes(image_path, output):
    # קריאת התמונה
    image = cv2.imread(image_path)

    # ציור תיבות הגבול
    for bbox in output["bboxes"]:
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # תיבה ירוקה

    # ציור תיבה מאוחדת אם קיימת
    if output["united_bbox"]:
        ux, uy, uw, uh = output["united_bbox"]
        cv2.rectangle(image, (ux, uy), (ux + uw, uy + uh), (255, 0, 0), 2)  # תיבה כחולה

    # הצגת התמונה
    cv2.imshow("Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_landmarks(image_path, output):
    # קריאת התמונה
    image = cv2.imread(image_path)

    # ציור נקודות ציון
    for hand_landmarks in output["hand_landmarks"]:
        for point in hand_landmarks:
            x = int(point["x"] * image.shape[1])
            y = int(point["y"] * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # נקודה אדומה

    # הצגת התמונה
    cv2.imshow("Hand Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
