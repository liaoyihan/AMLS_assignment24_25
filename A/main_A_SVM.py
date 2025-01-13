import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# **Step 1: Load the dataset**
# Define the file path to the dataset
file_path = os.path.abspath("./Datasets/breastmnist.npz")   # Get absolute path to the dataset
data = np.load(file_path) # Load the .npz dataset into a structured numpy object

# **Step 2: Extract data from the dataset**
# Extract training, validation, and test images
train_images = data['train_images']  # Extract the array of training images from the dataset
val_images = data['val_images']      # Extract the array of validation images from the dataset
test_images = data['test_images']    # Extract the array of test images from the dataset

# Extract corresponding labels and flatten them (convert from 2D to 1D arrays)
train_labels = data['train_labels'].ravel()   # Convert training labels to a 1D array
val_labels = data['val_labels'].ravel()       # Convert validation labels to a 1D array
test_labels = data['test_labels'].ravel()     # Convert test labels to a 1D array

# **Step 3: Preprocess the image data**
# Flatten the images (e.g., 28x28 images are converted to 784 feature vectors per image)
X_train = train_images.reshape(train_images.shape[0], -1)
X_val = val_images.reshape(val_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)

# Normalize the pixel values to have zero mean and unit variance
# StandardScaler scales each feature to a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Fit scaler on training data and transform it
X_val = scaler.transform(X_val)  # Transform validation data using the same scaler
X_test = scaler.transform(X_test)  # Transform test data using the same scaler

# **Step 4: Initialize and train the SVM model**
# Create an SVM classifier with a linear kernel
# The kernel type determines how the data is mapped into a higher-dimensional space
svm_model = SVC(kernel='linear', random_state=42) # Linear kernel SVM, random seed for reproducibility
# Train the SVM model on the training data
svm_model.fit(X_train, train_labels)

# **Step 5: Evaluate the model**
# Predict labels for the validation set
val_predictions = svm_model.predict(X_val)

# Predict labels for the test set
test_predictions = svm_model.predict(X_test)

# **Step 6: Display results**
# Print classification report for validation data
print("Validation Set Results:")
# Display a header indicating that the following results are for the validation dataset
print(classification_report(val_labels, val_predictions))

# Print classification report for test data
print("Test Set Results:")
# Display a header indicating that the following results are for the test dataset
print(classification_report(test_labels, test_predictions))

# Calculate and display accuracy for validation and test sets
val_accuracy = accuracy_score(val_labels, val_predictions)  # Compute validation accuracy
test_accuracy = accuracy_score(test_labels, test_predictions) # Compute test accuracy
print(f"Validation Accuracy: {val_accuracy:.4f}") # Display validation accuracy
print(f"Test Accuracy: {test_accuracy:.4f}") # Display test accuracy
