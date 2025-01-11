import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.regularizers import l2
import json
import pandas as pd

# **1. Data Reading**
# This function loads a dataset from a specified `.npz` file path.
# It returns the loaded dataset as a numpy object.
def Dataread(file_path):
    dataset = np.load(file_path)
    return dataset

# Function to split and categorize the dataset into training, validation, and test sets.
# Prints the shape of each data subset for verification.
def category_Data(dataset):
    train_images = dataset['train_images']  # Extract training images
    train_labels = dataset['train_labels']  # Extract training labels
    val_images = dataset['val_images']      # Extract validation images
    val_labels = dataset['val_labels']      # Extract validation labels
    test_images = dataset['test_images']    # Extract test images
    test_labels = dataset['test_labels']    # Extract test labels

    # Print the dimensions of each data subset to confirm the integrity of the dataset
    print("Train Images Shape:", train_images.shape)
    print("Train Labels Shape:", train_labels.shape)
    print("\nValidation Images Shape:", val_images.shape)
    print("Validation Labels Shape:", val_labels.shape)
    print("\nTest Images Shape:", test_images.shape)
    print("Test Labels Shape:", test_labels.shape)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

# **2. Image Display**
# Function to randomly display a subset of images from the dataset along with their labels.
# It selects random images, displays them in a grid, and saves the resulting plot.
def display_random_images(images, labels, num_images=5):
       # Select `num_images` random indices from the dataset
    random_indices = random.sample(range(len(images)), num_images)
        # Create a plot grid with `num_images` columns
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 3))
    for i, index in enumerate(random_indices):
        ax = axes[i]
        ax.imshow(images[index], cmap='gray') # Display image in grayscale
        ax.axis('off')  # Remove axes for cleaner visualization
        ax.set_title(f'Label: {labels[index]}') # Set the title as the label
        
    plt.suptitle("Sample of dataset Images", fontsize=16)# Add a main title to the plot
    plt.savefig("images/task_A/sample.png") # Save the plot to a file
    # plt.show()
    
# **3. Data Augmentation**
# Function to perform data augmentation on the training dataset.
# Augments images by applying transformations like flipping, rotation, brightness adjustment, and blurring.
def enhanced_data(train_images, train_labels):
    # Helper function to augment a single image
    def augment_image(image):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
        # Random rotation within the range [-30, 30] degrees
        rotation_angle = random.randint(-30, 30)
        image = image.rotate(rotation_angle)
        enhancer = ImageEnhance.Brightness(image)
        brightness_factor = random.uniform(0.8, 1.2) # Brightness scale factor
        image = enhancer.enhance(brightness_factor)
        
        # Random Gaussian blur with a 30% chance
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))
        return image
    
    # Helper function to augment the entire dataset
    def augment_and_merge_dataset(images, labels, num_augmentations=5):
        augmented_images = []  # List to store augmented images
        augmented_labels = []  # List to store corresponding labels
        for i in range(len(images)):
            original_image = Image.fromarray(images[i])  # Convert numpy array to PIL image
            
            # Add the original image to the augmented dataset
            augmented_images.append(images[i])
            augmented_labels.append(labels[i])
            
            # Generate `num_augmentations` augmented versions of the original image
            for _ in range(num_augmentations):
                augmented_image = augment_image(original_image)
                augmented_images.append(np.array(augmented_image)) # Convert back to numpy array
                augmented_labels.append(labels[i]) # Keep the same label
        return np.array(augmented_images), np.array(augmented_labels)

    # Apply augmentation to the training dataset
    augmented_train_images, augmented_train_labels = augment_and_merge_dataset(train_images, train_labels)
    #random plotting the image after enhanced
    num_samples = 5  # number of samples that needs to be show
    sample_indices = np.random.choice(augmented_train_images.shape[0], num_samples, replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for i, idx in enumerate(sample_indices):
        axes[i].imshow(augmented_train_images[idx])
        axes[i].set_title(f"Label: {augmented_train_labels[idx]}")
        axes[i].axis('off')
    # save the images
    plt.savefig("images/task_A/sample_augmented.png")
    # print the datasets check the size of the datasets
    print("\n")
    print("Augmented Train Images Shape:", augmented_train_images.shape)
    print("Augmented Train Labels Shape:", augmented_train_labels.shape)
    return augmented_train_images, augmented_train_labels
   
   
# **4. Normalization**
# This function normalizes the pixel values of the images to the range [0, 1].
# Normalization improves the performance and stability of the CNN during training.
def normalisation(train_images, val_images, test_images, augmented_train_images):
    train_images_normalized = train_images / 255.0
    augmented_train_images_normalized = augmented_train_images / 255.0
    val_images_normalized = val_images / 255.0
    test_images_normalized = test_images / 255.0

    # Print the min and max pixel values of each dataset to verify normalization
    print("\nNormalized Data Check:")
    print("Train Images - Min:", train_images_normalized.min(), "Max:", train_images_normalized.max())
    print("Augmented Train Images - Min:", augmented_train_images_normalized.min(), "Max:", augmented_train_images_normalized.max())
    print("Validation Images - Min:", val_images_normalized.min(), "Max:", val_images_normalized.max())
    print("Test Images - Min:", test_images_normalized.min(), "Max:", test_images_normalized.max())

    return train_images_normalized, augmented_train_images_normalized, val_images_normalized, test_images_normalized

# **5. CNN Training**
# This function builds, compiles, and trains a convolutional neural network (CNN) model.
# It also evaluates the model on the test dataset and plots the accuracy over epochs.
def train_model(train_images, train_labels, val_images, val_labels, test_images, test_labels, output_prefix):
    # L2 regularization strength to prevent overfitting
    l2_reg = 0.001
    
    # Define the CNN architecture using a Sequential model
    model = Sequential([
        # First convolutional layer with 5 filters, 3x3 kernel size, ReLU activation, and L2 regularization
        Conv2D(5, kernel_size=3, activation='relu', input_shape=(train_images.shape[1], train_images.shape[2], 1), kernel_regularizer=l2(l2_reg)),
        # Max pooling layer to reduce spatial dimensions
        MaxPooling2D(pool_size=(2, 2)),
        # Second convolutional layer with 9 filters, 3x3 kernel size, ReLU activation, and L2 regularization
        Conv2D(9, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_reg)),
        # Max pooling layer to further reduce spatial dimensions
        MaxPooling2D(pool_size=(2, 2)),
        # Flatten the output of the convolutional layers to prepare for dense layers
        Flatten(),
        # Fully connected (dense) layer with 32 units, ReLU activation, and L2 regularization
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        # Dropout layer with a 50% dropout rate to prevent overfitting
        Dropout(0.5),
        # Output layer with 1 unit and sigmoid activation for binary classification
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model using Adam optimizer, binary cross-entropy loss, and accuracy as a metric
     # Train the model on the training data and validate on the validation data
    # Number of epochs and batch size can be adjusted for better results
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=60, batch_size=32)   # change for epoch number for better accuracy
   
    # Evaluate the model on the test dataset and print the results
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"{output_prefix} Test Accuracy: {test_accuracy}")
    print(f"{output_prefix} Test Loss: {test_loss}")
    # Plot the training and validation accuracy over epochs
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()

    return history

# **6. Save Results to CSV**
# This function saves the training history (accuracy and loss for training and validation) to a CSV file.
def save_results_to_csv(history, csv_file):
    """
    Save training history results to a CSV file.
    
    Parameters:
        history (History): Training history object from Keras.
        csv_file (str): Path to the CSV file to save results.
    """
    # Extract the training history into a dictionary
    history_dict = {
        "epoch": list(range(1, len(history.history['accuracy']) + 1)), # Epoch numbers
        "accuracy": history.history['accuracy'], # Training accuracy per epoch
        "val_accuracy": history.history['val_accuracy'], # Validation accuracy per epoch
        "loss": history.history['loss'], # Training loss per epoch
        "val_loss": history.history['val_loss']  # Validation loss per epoch
    }
    # Convert the dictionary to a pandas DataFrame for easier manipulation and saving
    df = pd.DataFrame(history_dict)
    
    # Save the DataFrame to a CSV file without the index column
    df.to_csv(csv_file, index=False)  
    print(f"Results saved to {csv_file}")  # Confirm the file has been saved
    print(df)  # Display the DataFrame for a quick check of the saved results


# **7. Save and Plot Results**
# This function generates a plot showing the accuracy and loss for both training and validation over epochs.
# The plot is saved to a specified file for later review.
def plot_results(history, output_file, title):
    """
    Plot training and validation accuracy/loss over epochs and save the plot.

    Parameters:
        history (History): Training history object from Keras containing metrics like accuracy and loss.
        output_file (str): Path to save the generated plot.
        title (str): Custom title for the overall plot.
    """
    # Extract metrics from the training history
    epochs = range(1, len(history.history['accuracy']) + 1)  # Epoch numbers
    accuracy = history.history['accuracy']  # Training accuracy
    val_accuracy = history.history['val_accuracy']  # Validation accuracy
    loss = history.history['loss']  # Training loss
    val_loss = history.history['val_loss']  # Validation loss
    
    # Set up the figure with a 1x2 grid for accuracy and loss plots
    plt.figure(figsize=(12, 6))
    
    # Accuracy plot (left subplot)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, label="Training Accuracy") # Plot training accuracy
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")  # Plot validation accuracy
    plt.title("Accuracy Over Epochs")  # Add a title to the accuracy plot
    plt.xlabel("Epochs") # Label for x-axis
    plt.ylabel("Accuracy") # Label for y-axis
    plt.legend() # Add legend to distinguish training/validation lines
    
    # Loss plot (right subplot)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Training Loss") # Plot training loss
    plt.plot(epochs, val_loss, label="Validation Loss") # Plot validation loss
    plt.title("Loss Over Epochs") # Add a title to the loss plot
    plt.xlabel("Epochs")  # Label for x-axis
    plt.ylabel("Loss") # Label for y-axis
    plt.legend() # Add legend to distinguish training/validation lines
    
    
    # Add a custom title for the entire plot
    plt.suptitle(title, fontsize=16) # Larger font for the main title
    
    # Save the plot
    plt.tight_layout() # Adjust layout to prevent overlapping
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")  # Confirm the file has been saved
    # Close the plot to release memory
    plt.close()



# **Main Execution**
if __name__ == "__main__":
    os.makedirs("images/task_A", exist_ok=True)  # Create `images/` folder for saving visualizations
    os.makedirs("results/task_A_result", exist_ok=True)  # Create a results folder for CSV files
    
    # Define the path to the dataset
    file_path = "./Datasets/breastmnist.npz" # Path to the dataset file in `.npz` format
    dataset = Dataread(file_path) # Load dataset into a structured numpy object
    
    # Step 2: Split the dataset into training, validation, and testing subsets
    train_images, train_labels, val_images, val_labels, test_images, test_labels = category_Data(dataset)
    
    # Step 3: Display a random subset of training images along with their labels
    display_random_images(train_images, train_labels)
    
    # Step 4: Perform data augmentation on the training dataset
    augmented_train_images, augmented_train_labels = enhanced_data(train_images, train_labels)
    
    # Step 5: Normalize all datasets to bring pixel values into the range [0, 1]
    train_images_normalized, augmented_train_images_normalized, val_images_normalized, test_images_normalized = normalisation(
        train_images, val_images, test_images, augmented_train_images
    )
    
    # Step 6: Train the CNN model using the original training data
    history_ori = train_model(
        train_images_normalized, train_labels, val_images_normalized, val_labels, test_images_normalized, test_labels, "original"
    )
    # Save the training results from the original data
    save_results_to_csv(history_ori, "results/task_A_result/results_table_original.csv")
    plot_results(history_ori, "images/task_A/original_training_plot.png", "Training and Validation Metrics by Original Data")
    
    # Train on augmented data and save results
    history_aug = train_model(
        augmented_train_images_normalized, augmented_train_labels, val_images_normalized, val_labels, test_images_normalized, test_labels, "augmented"
    )
    # Save the training results from the augmented data
    save_results_to_csv(history_aug, "results/task_A_result/results_table_augmented.csv")
    plot_results(history_aug, "images/task_A/augmented_training_plot.png", "Training and Validation Metrics by Augmented Data")
