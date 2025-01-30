# Main file for execution

from src.process_image import process_image
from src.visualize import visualize_bboxes, visualize_landmarks
import json

if __name__ == "__main__":

    # List of image paths to process
    image_paths = [
        "data\\grip3.jpg",
        "data\\grip4.jpg",
        "data\\grip5.jpg",
        "data\\grip6.jpg",
        "data\\grip7.jpg",
        "data/fingerpicture.png",
        "data\\palm.jpeg",
        "data\\me1.jpg",
        "data\\me2.jpg",
        "data\\me3.jpg",
        "data\\me4.jpg",
        "data\\hodi.jpg",
        "data\\dislike.jpeg",
        "data\\grip1.png",
        "data\\like.jpeg",
        "data\\noGesture.jpeg",
        "data\\like_and_dislike.jpeg",
        "data\\point.jpeg",
        "data\\dislike2.jpeg",
        "data\\grip.png"
    ]

    # Loop through each image in the list
    for image in image_paths:
        # Process the image
        data = process_image(image)
        
        # Visualize and display bounding boxes on the image
        visualize_bboxes(image, data)
        
        # Visualize and display hand landmarks on the image
        visualize_landmarks(image, data)

        # Save the processed result as a JSON file
        output_path = f"data/output/result_{image[5:image.index('.')]}.json"
        with open(output_path, "w") as f:
            json.dump({"temp": data}, f, indent=4)

        print(f"Processed image saved to {output_path}")
