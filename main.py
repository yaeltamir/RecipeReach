# קובץ ראשי להרצה

from src.process_image import process_image
import json

if __name__ == "__main__":
    # מסלול לתמונה לדוגמה
    #image_path = "C:/Users/ACER/Documents/לימודים/שנה ג/סמסטר א/בינה מלאכותית/RecipeReach/RecipeReach/data/lovepik-clap-a-personal-boys-hand-clicking-png-image_400760113_wh1200.png"
    image_path = r"C:\Users\ACER\Documents\לימודים\שנה ג\סמסטר א\בינה מלאכותית\RecipeReach\RecipeReach\data\fingerpicture.png"

    # עיבוד התמונה
    data = process_image(image_path)
    print(data)

    # שמירת התוצאה כקובץ JSON
    output_path = "data/output/result.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Processed image saved to {output_path}")
