import cv2
import mediapipe as mp
import uuid

def process_image(image_frame):
    """
    Processes an input image to detect hands, extract bounding boxes, and return relevant hand landmarks.
    
    :param image_frame: Input image as a NumPy array (BGR format)
    :return: Dictionary containing bounding boxes, unified bounding box (if two hands detected),
             hand landmarks, and metadata (age, gender, race - placeholders for future implementation)
    """
    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    
    # Get image dimensions
    height, width, _ = image_frame.shape
    
    # Convert image to RGB and process with MediaPipe
    results = hands.process(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB))
    
    # Output dictionary to store results
    output = {
        "bboxes": [],  # List of bounding boxes for detected hands
        "united_bbox": None,  # Unified bounding box if two hands are detected
        "user_id": str(uuid.uuid4()),  # Unique user identifier (randomly generated)
        "hand_landmarks": [],  # List of hand landmarks for each detected hand
        "meta": {
            "age": None,  # Placeholder for future age estimation
            "gender": None,  # Placeholder for future gender detection
            "race": None  # Placeholder for future race classification
        }
    }
    
    # Process detected hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Compute bounding box coordinates
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * width
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * height
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * width
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * height
            
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            output["bboxes"].append(bbox)
            
            # Store hand landmarks
            landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            output["hand_landmarks"].append(landmarks)
        
        # Compute a unified bounding box if two hands are detected
        if len(output["bboxes"]) == 2:
            united_x_min = min(output["bboxes"][0][0], output["bboxes"][1][0])
            united_y_min = min(output["bboxes"][0][1], output["bboxes"][1][1])
            united_x_max = max(output["bboxes"][0][0] + output["bboxes"][0][2],
                               output["bboxes"][1][0] + output["bboxes"][1][2])
            united_y_max = max(output["bboxes"][0][1] + output["bboxes"][0][3],
                               output["bboxes"][1][1] + output["bboxes"][1][3])
            
            output["united_bbox"] = [int(united_x_min), int(united_y_min),
                                     int(united_x_max - united_x_min), int(united_y_max - united_y_min)]
    
    # Close MediaPipe Hands instance
    hands.close()
    
    return output
