# קובץ ראשי להרצה

from src.process_image import process_image
from src.visualize import visualize_bboxes, visualize_landmarks
import json

if __name__ == "__main__":
    # מסלול לתמונה לדוגמה

    image_path="data/fingerpicture.png"
    # image_path="data\\palm.jpeg"
    # image_path="data\\me1.jpg"
    # image_path="data\\me2.jpg"
    # image_path="data\\me3.jpg"
    #image_path="data\me4.jpg"
    #image_path="data\me5.jpg"
    #image_path="PATH"
    # עיבוד התמונה
    data = process_image(image_path)
    # הצגת Bounding Boxes
    visualize_bboxes(image_path, data)

    # הצגת נקודות ציון
    visualize_landmarks(image_path, data)
   # print(data)

    # שמירת התוצאה כקובץ JSON
    output_path = "data/output/result.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Processed image saved to {output_path}")
