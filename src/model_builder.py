import keras as ker
import numpy as np
from data_loader import load_data_from_files
from sklearn.utils.class_weight import compute_class_weight

# Define training, testing, and validation dataset file paths
datasetTrain_files=[
    'dataset/train/like_train.json',
    'dataset/train/dislike_train.json',
    'dataset/train/palm_train.json',
    'dataset/train/grip_train.json',
    'dataset/train/point_train.json',
    'dataset/train/no_gesture_train.json'
]

datasetTest_files=[
    'dataset/test/like_test.json',
    'dataset/test/dislike_test.json',
    'dataset/test/palm__test.json',
    'dataset/test/grip_test.json',
    'dataset/test/point_test.json',
    'dataset/test/no_gesture__test.json'
]

datasetValidate_files = [
    'dataset/like.json',
    'dataset/dislike.json',
    'dataset/palm.json',
    'dataset/grip.json',
    'dataset/point.json',
    'dataset/no_gesture.json'
]


def define_model(input_shape, num_classes):
    """
    Defines and builds a neural network model for gesture classification.
    
    :param input_shape: Shape of the input data (sequence length, feature dimension)
    :param num_classes: Number of output classes for classification
    :return: A compiled Keras sequential model
    """
    model = ker.models.Sequential()
    
    # Masking layer to ignore specific padding values
    model.add(ker.layers.Masking(mask_value=[-2.0, -2.0], input_shape=input_shape))  
    
    # Flatten input into a single vector
    model.add(ker.layers.Flatten(input_shape=input_shape))  
    
    # Fully connected layers with dropout for regularization
    model.add(ker.layers.Dense(256, activation='relu'))  
    model.add(ker.layers.Dropout(0.4))
    
    model.add(ker.layers.Dense(128, activation='relu'))  
    model.add(ker.layers.Dropout(0.35))
    
    model.add(ker.layers.Dense(64, activation='relu'))  
    model.add(ker.layers.Dropout(0.3))
    
    model.add(ker.layers.Dense(32, activation='relu'))  
    
    # Output layer with softmax activation for multi-class classification
    model.add(ker.layers.Dense(num_classes, activation='softmax'))  
    
    return model


def prepare_datasets(num_categories):
    """
    Loads and prepares training, validation, and testing datasets.
    
    :param num_categories: Number of classification categories
    :return: Dictionaries containing feature and label sets for training, validation, and testing
    """
    # Load training data
    features_train, labels_train = load_data_from_files(datasetTrain_files)
    print("Training features shape:", features_train.shape)
    labels_train = ker.utils.to_categorical(labels_train, num_classes=num_categories)
    print("Training labels shape:", labels_train.shape)

    # Load testing data
    features_test, labels_test = load_data_from_files(datasetTest_files, features_train.shape[1])
    print("Testing features shape:", features_test.shape)
    labels_test = ker.utils.to_categorical(labels_test, num_classes=num_categories)
    print("Testing labels shape:", labels_test.shape)

    # Load validation data
    features_validate, labels_validate = load_data_from_files(datasetValidate_files, features_train.shape[1])
    print("Validation features shape:", features_validate.shape)
    labels_validate = ker.utils.to_categorical(labels_validate, num_classes=num_categories)
    print("Validation labels shape:", labels_validate.shape)

    return {'features': features_train, 'labels': labels_train}, \
           {'features': features_validate, 'labels': labels_validate}, \
           {'features': features_test, 'labels': labels_test}


def build_model(model_name, num_categories, train_set, test_set, num_epochs=10, size_batch=32):
    """
    Builds, compiles, and trains a neural network model.

    :param model_name: Name to save the trained model
    :param num_categories: Number of classification categories
    :param train_set: Training dataset containing 'features' and 'labels'
    :param test_set: Testing dataset containing 'features' and 'labels'
    :param num_epochs: Number of training iterations (default: 10)
    :param size_batch: Batch size for training (default: 32)
    :return: None (saves the trained model)
    """
    input_shape = (train_set['features'].shape[1], train_set['features'].shape[2])
    model = define_model(input_shape, num_categories)

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Compute class weights to handle imbalanced data
    class_weights = compute_class_weight(
        class_weight='balanced',  
        classes=np.unique(np.argmax(train_set['labels'], axis=1)),
        y=np.argmax(train_set['labels'], axis=1)
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("Class Weights:", class_weight_dict)

    # Train the model
    model.fit(train_set['features'], train_set['labels'], 
              epochs=num_epochs, batch_size=size_batch, 
              validation_data=(test_set['features'], test_set['labels']), 
              class_weight=class_weight_dict)

    # Save the trained model
    model.save(f'{model_name}.keras')
