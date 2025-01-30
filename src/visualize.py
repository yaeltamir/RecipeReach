import cv2

def visualize_bboxes(image_path, output):
    """
    Function to visualize bounding boxes on an image.

    Args:
        image_path: Path to the image file.
        output: A dictionary containing the bounding boxes ("bboxes") and the unified bounding box ("united_bbox").
            - "bboxes" is a list of bounding boxes where each bounding box is represented by [x, y, w, h].
            - "united_bbox" is a single unified bounding box in the same format, represented by [ux, uy, uw, uh].

    Displays the image with green bounding boxes for individual detections and a blue bounding box for the unified box.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Draw the bounding boxes
    for bbox in output["bboxes"]:
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

    # Draw the unified bounding box if it exists
    if output["united_bbox"]:
        ux, uy, uw, uh = output["united_bbox"]
        cv2.rectangle(image, (ux, uy), (ux + uw, uy + uh), (255, 0, 0), 2)  # Blue box

    # Display the image
    cv2.imshow("Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_landmarks(image_path, output):
    """
    Function to visualize hand landmarks on an image.

    Args:
        image_path: Path to the image file.
        output: A dictionary containing the hand landmarks ("hand_landmarks").
            - "hand_landmarks" is a list of landmarks for each hand, where each landmark is a list of [x, y] points.

    Displays the image with red circles marking the hand landmarks.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Draw the hand landmarks
    for hand_landmarks in output["hand_landmarks"]:
        for point in hand_landmarks:
            x = int(point[0] * image.shape[1])
            y = int(point[1] * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red dot

    # Display the image
    cv2.imshow("Hand Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
