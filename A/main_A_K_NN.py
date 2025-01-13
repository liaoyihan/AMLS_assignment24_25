import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# **1. Data Loading**
def load_data(file_path):
    """
    Load the dataset from an .npz file.
    
    Parameters:
        file_path (str): Path to the .npz dataset.
    
    Returns:
        Tuple: Training, validation, and test images and labels.
    """
    # Load the dataset from the provided .npz file
    dataset = np.load(file_path)
    train_images = dataset['train_images']  # Training images (e.g., shape: (num_samples, height, width))
    train_labels = dataset['train_labels']  # Training labels corresponding to the training images
    val_images = dataset['val_images']  # Validation images used to evaluate the model during training
    val_labels = dataset['val_labels']  # Validation labels corresponding to the validation images
    test_images = dataset['test_images']  # Test images used to evaluate the final performance of the model
    test_labels = dataset['test_labels']  # Test labels corresponding to the test images
    # Return the extracted training, validation, and test datasets
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

# **2. Data Preprocessing**
def preprocess_data(train_images, val_images, test_images):
    """
    Preprocess the image data by flattening and normalizing.
    
    Parameters:
        train_images, val_images, test_images (np.ndarray): Input image datasets.
    
    Returns:
        Tuple: Preprocessed training, validation, and test data.
    """
    # Flatten the images (28x28 -> 784)
    train_images_flattened = train_images.reshape(train_images.shape[0], -1)
    val_images_flattened = val_images.reshape(val_images.shape[0], -1)
    test_images_flattened = test_images.reshape(test_images.shape[0], -1)
    
    # Normalize pixel values to have zero mean and unit variance (using StandardScaler)
    scaler = StandardScaler() # Initialize the scaler
    # Fit the scaler on training data and transform it
    train_images_normalized = scaler.fit_transform(train_images_flattened)
    # Apply the same transformation to validation and test datasets
    val_images_normalized = scaler.transform(val_images_flattened)
    test_images_normalized = scaler.transform(test_images_flattened)
    
    # Return normalized datasets
    return train_images_normalized, val_images_normalized, test_images_normalized

# **3. Train and Evaluate K-NN**
def train_and_evaluate_knn(train_images, train_labels, val_images, val_labels, test_images, test_labels, k=5):
    """
    Train and evaluate the K-NN classifier.
    
    Parameters:
        train_images, val_images, test_images (np.ndarray): Preprocessed image data.
        train_labels, val_labels, test_labels (np.ndarray): Corresponding labels.
        k (int): Number of neighbors for K-NN.
    
    Returns:
        None
    """
    # Initialize the K-NN model with the specified number of neighbors (k)
    knn_model = KNeighborsClassifier(n_neighbors=k)
    
    # Train the K-NN model on the training data
    knn_model.fit(train_images, train_labels)
    
    # Evaluate the model on the validation set
    val_predictions = knn_model.predict(val_images)  # Predict validation labels
    val_accuracy = accuracy_score(val_labels, val_predictions)  # Calculate validation accuracy
    print(f"Validation Accuracy: {val_accuracy:.4f}") # Display validation accuracy
    
    # Print a detailed classification report for validation predictions
    print("\nValidation Classification Report:")
    print(classification_report(val_labels, val_predictions))
    
    # Evaluate the model on the test set
    test_predictions = knn_model.predict(test_images) # Predict test labels
    test_accuracy = accuracy_score(test_labels, test_predictions) # Calculate test accuracy
    print(f"Test Accuracy: {test_accuracy:.4f}") # Display test accuracy
    
    # Print a detailed classification report for test predictions
    print("\nTest Classification Report:")
    print(classification_report(test_labels, test_predictions))

# **Main Execution**
if __name__ == "__main__":
    # File path to the dataset
    file_path = "./Datasets/breastmnist.npz"  # Path to the dataset in .npz format
    
    # Step 1: Load data from the dataset file
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data(file_path)
    
    # Step 2: Preprocess the data by flattening and normalizing
    train_images_normalized, val_images_normalized, test_images_normalized = preprocess_data(
        train_images, val_images, test_images
    )
    
    # Step 3: Train the K-NN model and evaluate its performance
    train_and_evaluate_knn(
        train_images_normalized, train_labels,  # Training data and labels
        val_images_normalized, val_labels, # Validation data and labels
        test_images_normalized, test_labels, # Test data and labels
        k=5  # Number of neighbors for K-NN; can be adjusted for experimentation
    )
