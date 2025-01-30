import keras as ker
import numpy as np
from data_loader import load_data_from_files
from sklearn.utils.class_weight import compute_class_weight


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
    Builds a neural network model.
    
    :param input_shape: Shape of the input (length of the data vector)
    :param num_classes: Number of classification categories
    :return: The defined model
    """
    model = ker.models.Sequential()  # Creates a sequential model where layers are added one after another
    
    # Ignores padding values added to the input; in this case, the mask values are set to [-2.0, -2.0]
    model.add(ker.layers.Masking(mask_value=[-2.0, -2.0], input_shape=input_shape))  
    
    # Flattens the input into a single vector
    model.add(ker.layers.Flatten(input_shape=input_shape))  
    
    # Fully connected layer with 256 neurons
    model.add(ker.layers.Dense(256, activation='relu'))  
    model.add(ker.layers.Dropout(0.4))  # Dropout to reduce overfitting (randomly disables 40% of neurons)
    
    # ReLU returns the input value if positive or 0 if negative
    model.add(ker.layers.Dense(128, activation='relu'))  
    model.add(ker.layers.Dropout(0.35))  # Dropout layer (35% of neurons are randomly disabled)
    
    model.add(ker.layers.Dense(64, activation='relu'))  
    model.add(ker.layers.Dropout(0.3))  # Dropout layer (30% of neurons are randomly disabled)
    
    model.add(ker.layers.Dense(32, activation='relu'))  
    
    # Output layer for classification
    # The last layer has as many neurons as the number of classification categories
    # Softmax converts the output into a probability vector summing to 1
    model.add(ker.layers.Dense(num_classes, activation='softmax'))  
    
    return model

  

def prepare_datasets(num_categories):
    #train section
    features_train,labels_train=load_data_from_files(datasetTrain_files)
    print("features train:(number of sumples, num of attributes,number of inner attributes) =", features_train.shape) 
    # num_categories=len(np.unique(labels_train))
    # print("num of different labels:",num_categories)
    labels_train = ker.utils.to_categorical(labels_train, num_classes=num_categories)
    print("labels train:(number of sumples, num of attributes) =", labels_train.shape)

    #test section
    features_test,labels_test=load_data_from_files(datasetTest_files,features_train.shape[1])
    print("features test:(number of sumples, num of attributes,number of inner attributes) =", features_test.shape) 
    labels_test = ker.utils.to_categorical(labels_test, num_classes=num_categories)
    print("labels test:(number of sumples, num of attributes) =", labels_test.shape)

    #validate section
    features_validate,labels_validate=load_data_from_files(datasetValidate_files,features_train.shape[1])
    print("features validate:(number of sumples, num of attributes,number of inner attributes) =", features_validate.shape) 
    labels_validate = ker.utils.to_categorical(labels_validate, num_classes=num_categories)
    print("labels validate:(number of sumples, num of attributes) =", labels_validate.shape)

    return {'features':features_train,'labels':labels_train},\
           {'features':features_validate,'labels':labels_validate},\
           {'features':features_test,'labels':labels_test}


def build_model(model_name, num_categories, train_set, test_set, num_epochs=10, size_batch=32):
    """
    Builds, compiles, and trains a neural network model.

    :param model_name: Name for saving the trained model.
    :param num_categories: Number of classification categories.
    :param train_set: Training dataset containing 'features' and 'labels'.
    :param test_set: Testing dataset containing 'features' and 'labels'.
    :param num_epochs: Number of training iterations (default is 10).
    :param size_batch: Batch size for training (default is 32).
    :return: None (saves the trained model).
    """
    input_shape = (train_set['features'].shape[1], train_set['features'].shape[2])
    model = define_model(input_shape, num_categories)

    """
    optimizer='adam': Adaptive optimization algorithm combining Momentum and RMSprop, automatically updating weights during training.

    loss='categorical_crossentropy': Loss function for multi-class classification, measuring the error between predicted and actual labels.

    metrics=['accuracy']: Evaluation metric that calculates the percentage of correctly classified samples.
    """
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',  # Automatically balances weights based on sample distribution
        classes=np.unique(np.argmax(train_set['labels'], axis=1)),  # Unique class labels
        y=np.argmax(train_set['labels'], axis=1)  # Class labels
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




    

    



