import json
import numpy as np

# Mapping of gesture labels to numerical values
label_mapping = {'palm': 0, 'point': 1, 'grip': 2, 'like': 3, 'dislike': 4, 'no_gesture': 5}

def load_data_from_files(file_paths, maxTarget=-1):
    """
    Loads data from multiple JSON files and combines all examples into X and Y arrays.

    :param file_paths: List of file paths to load data from.
    :param maxTarget: The maximum number of hand keypoints; if set to -1, it is determined automatically.
    :return: NumPy arrays X (features) and Y (labels).
    """
    X, Y = [], []

    if maxTarget == -1:
        maxTarget = get_max_length(file_paths)

    for file_path in file_paths:
        x_data, y_data = load_data(file_path, maxTarget)
        X.extend(x_data)
        Y.extend(y_data)

    return np.array(X), np.array(Y)

def get_max_length(dataset_files):
    """
    Determines the maximum number of hand keypoints from a list of dataset files.

    :param dataset_files: List of file paths containing hand landmark data.
    :return: Maximum length of flattened hand landmark coordinates.
    """
    max_length = 0

    for file_path in dataset_files:
        with open(file_path, 'r') as file:
            data = json.load(file)

            for obj in data.values():
                hand_landmarks = obj['hand_landmarks']
                flat_landmarks = [coordinate for landmark in hand_landmarks for coordinate in landmark]
                max_length = max(max_length, len(flat_landmarks))

    return max_length

def flatten_Coordinates(hand_landmarks, target_length):
    """
    Flattens hand landmark coordinates into a fixed-size array of length `target_length`.
    If there are too few points, padding with (-2.0, -2.0) is added.
    If there are too many points, extra ones are truncated.

    :param hand_landmarks: List of hand landmarks (each with x, y coordinates).
    :param target_length: Desired length of the output array.
    :return: Numpy array with standardized length.
    """
    hand_landmarks = np.array(hand_landmarks)  # Convert to NumPy array

    if len(hand_landmarks) < target_length:
        # Padding with (-2.0, -2.0) if there are not enough points
        padding = [[-2.0, -2.0]] * (target_length - len(hand_landmarks))
        hand_landmarks = np.vstack((hand_landmarks, padding))
    
    elif len(hand_landmarks) > target_length:
        # Truncate extra points if there are too many
        hand_landmarks = hand_landmarks[:target_length]

    return hand_landmarks

def load_data(filename, maxTarget=21, istest=False):
    """
    Reads a JSON file and prepares the data for model training.

    :param filename: Name of the JSON file to load data from.
    :param maxTarget: The desired number of hand keypoints.
    :param istest: Boolean flag indicating if the data is for testing (default is False).
    :return: NumPy arrays of features (X) and labels (Y).
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    X = []
    Y = []

    for obj in data.values():
        hand_landmarks = obj['hand_landmarks'][0]  # Extract the first set of hand landmarks

        if len(hand_landmarks) == 0:
            continue
        
        if not istest:
            label = obj['labels']  # Retrieve the label
            # Convert the classification label into a numerical value (e.g., "like" → 3)
            Y.append(label_mapping[label[0]])

        # Ensure a uniform size for the number of hand keypoints
        flattened_landmarks = flatten_Coordinates(hand_landmarks, maxTarget)
        X.append(flattened_landmarks)

    return np.array(X), np.array(Y)
