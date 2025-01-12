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
    # Load the dataset from the specified .npz file
    dataset = np.load(file_path)
    
    # Extract training, validation, and test subsets
    train_images = dataset['train_images']  # Training images (e.g., shape: (num_samples, height, width))
    train_labels = dataset['train_labels']  # Training labels corresponding to the training images
    val_images = dataset['val_images']      # Validation images used to monitor model performance during training
    val_labels = dataset['val_labels']      # Validation labels corresponding to the validation images
    test_images = dataset['test_images']    # Test images for evaluating the final model performance
    test_labels = dataset['test_labels']    # Test labels corresponding to the test images
    
    # Return the data as separate variables
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
    
    # Normalize pixel values to [0, 1]
    scaler = StandardScaler()
    train_images_normalized = scaler.fit_transform(train_images_flattened)
    val_images_normalized = scaler.transform(val_images_flattened)
    test_images_normalized = scaler.transform(test_images_flattened)
    
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
    # Train the K-NN model
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_images, train_labels)
    
    # Validate the model
    val_predictions = knn_model.predict(val_images)  # Predict validation labels
    val_accuracy = accuracy_score(val_labels, val_predictions)  # Calculate validation accuracy
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(val_labels, val_predictions))  # Print detailed classification report
    
    # Test the model
    test_predictions = knn_model.predict(test_images)   # Predict test labels
    test_accuracy = accuracy_score(test_labels, test_predictions)   # Calculate test accuracy
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(test_labels, test_predictions))   # Print detailed classification report

# **Main Execution**
if __name__ == "__main__":
    # File path to the dataset
    file_path = "./Datasets/bloodmnist.npz"  # Change to your dataset path
    
    # Step 1: Load data
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data(file_path)
    
    # Step 2: Preprocess data
    train_images_normalized, val_images_normalized, test_images_normalized = preprocess_data(
        train_images, val_images, test_images
    )
    
    # Step 3: Train and evaluate K-NN
    train_and_evaluate_knn(
        train_images_normalized, train_labels,
        val_images_normalized, val_labels,
        test_images_normalized, test_labels,
        k=5  # Change k value if needed
    )
