import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt



#function to train the model
def train_model(train_images_normalized, val_images_normalized, test_images_normalized, train_labels, val_labels, test_labels):
    
    # change the label to one hot code
    train_labels_cat = to_categorical(train_labels, num_classes=9)
    val_labels_cat = to_categorical(val_labels, num_classes=9)
    test_labels_cat = to_categorical(test_labels, num_classes=9)
    # creat the model
    model = Sequential([
        #hidden layer one with input layer of 28*28*3
        Conv2D(32,  # Number of filters in the convolutional layer
               kernel_size=(3, 3),  # Size of the filter kernel (3x3 pixels)
               activation='relu',    # Use ReLU activation to introduce non-linearity
               input_shape=(28, 28, 3),  # Input shape: 28x28 images with 3 channels (RGB)
               kernel_regularizer=l2(0.001)),  # L2 regularization to prevent overfitting
        MaxPooling2D(pool_size=(2, 2)),
        #hidden layer 2
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        #hidden layer 3
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        #hidden layer 4
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.25),
        #output
        Dense(9, activation='softmax')

    ])

    # Compile the model with Adam optimizer and categorical crossentropy
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model and validate on validation data
    history = model.fit(train_images_normalized, train_labels_cat, batch_size=32, epochs=50,      # change the epoch number for training
                        validation_data=(val_images_normalized, val_labels_cat))

    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate( test_images_normalized, test_labels_cat)
    print("Test Loss:", test_loss)  # Print the final loss on the test dataset
    print("Test Accuracy:", test_accuracy)   # Print the final accuracy on the test dataset

    # Save epoch-wise results and plot training progress
    save_epoch_results(history, test_loss, test_accuracy, "CNN_Custom")  # Save results and generate a plot of training progress
    
        # Plotting the accuracy and loss graphs side by side
    plt.figure(figsize=(12, 4))  # Set the figure size (width: 12, height: 4)

    # Plot accuracy over epochs
    plt.subplot(1, 2, 1)   # Create the first subplot (1 row, 2 columns, position 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')  # Plot training accuracy
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Plot validation accuracy
    plt.title('Accuracy over Epochs')  # Add a title to the plot
    plt.xlabel('Epoch')  # Label for the x-axis
    plt.ylabel('Accuracy')  # Label for the y-axis
    plt.legend()  # Add a legend to differentiate between training and validation accuracy

    # Plot loss over epochs
    plt.subplot(1, 2, 2)  # Create the second subplot (1 row, 2 columns, position 2)
    plt.plot(history.history['loss'], label='Train Loss')   # Plot training loss
    plt.plot(history.history['val_loss'], label='Validation Loss')   # Plot validation loss
    plt.title('Loss over Epochs')  # Add a title to the plot
    plt.xlabel('Epoch')  # Label for the x-axis
    plt.ylabel('Loss') # Label for the y-axis
    plt.legend()  # Add a legend to differentiate between training and validation loss

    plt.suptitle("The result of the Customised-CNN model",fontsize=16)   # Larger font size for emphasis
    plt.savefig("./images/task_B/Training and testing CNN training.png")   # Save the plot to a specified file path
    
    return history   # Return the training history object for further analysis



def save_epoch_results(history, test_loss, test_accuracy, model_name):
    """
    Save each epoch's training results and final test results to a CSV file.
    """
    # Prepare a dictionary containing epoch-wise training and validation metrics
    history_dict = {
        "epoch": list(range(1, len(history.history['accuracy']) + 1)),  # Epoch numbers
        "train_accuracy": history.history['accuracy'],  # Training accuracy per epoch
        "val_accuracy": history.history['val_accuracy'],   # Validation accuracy per epoch
        "train_loss": history.history['loss'],   # Training loss per epoch
        "val_loss": history.history['val_loss']  # Validation loss per epoch
    }
    # Convert the dictionary into a Pandas DataFrame for structured storage
    history_df = pd.DataFrame(history_dict)
    
    # Add test loss and accuracy as additional columns (constant across epochs)
    history_df["test_loss"] = test_loss
    history_df["test_accuracy"] = test_accuracy

    # Ensure the directory for saving results exists
    os.makedirs("results/task_B_result", exist_ok=True)
    # Define the file path for saving the CSV file
    results_file = f"results/task_B_result/{model_name.lower()}_epoch_results.csv"
    # Save the DataFrame to a CSV file
    history_df.to_csv(results_file, index=False)
    print(f"{model_name} training and test results saved to {results_file}")
    return history



# Function to train a ResNet50-based model
def CNN_resnet(train_images_normalized,train_labels,val_images_normalized,val_labels,test_images_normalized,test_labels):
    # Convert labels to one-hot encoding (for multi-class classification with 9 classes)
    train_labels_cat = to_categorical(train_labels, num_classes=9)
    val_labels_cat = to_categorical(val_labels, num_classes=9)
    test_labels_cat = to_categorical(test_labels, num_classes=9)
    # ensure the label is float 32 format
    train_images_normalized = train_images_normalized.astype('float32')
    val_images_normalized = val_images_normalized.astype('float32')
    test_images_normalized = test_images_normalized.astype('float32')

    # pre- process the images to satify resnet requirment
    train_images_prep = preprocess_input(train_images_normalized)
    val_images_prep = preprocess_input(val_images_normalized)
    test_images_prep = preprocess_input(test_images_normalized)
    
    # load resnet50 model
    base_model = ResNet50(weights='imagenet', # Use pre-trained weights from ImageNet
                include_top=False,   # Exclude the fully connected layers
                input_tensor=Input(shape=(28, 28, 3)))  # Specify the input shape (28x28 RGB images)

    # change the output layer to satisify task
    x = base_model.output  # Output from the ResNet50 base model
    x = GlobalAveragePooling2D()(x)  # Global average pooling to reduce feature map dimensions
    x = Dense(1024, activation='relu')(x)  # Fully connected layer with 1024 units and ReLU activation
    predictions = Dense(9, activation='softmax')(x)   # Output layer with 9 units (softmax for multi-class classification)

     # Create the complete model by combining base model and custom layers
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the layers of the base model to retain pre-trained features
    for layer in base_model.layers:
        layer.trainable = False  # Prevent updating weights of the pre-trained layers

    # Compile the model
    # Use Adam optimizer, categorical crossentropy as the loss function, and accuracy as the evaluation metric
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    history = model.fit(train_images_prep,  # Preprocessed training images
                        train_labels_cat,   # One-hot encoded training labels
                        batch_size=32,  # Number of samples per gradient update
                        epochs=50,     # Number of epochs to train the model (adjustable for experimentation)
                        validation_data=(val_images_prep, val_labels_cat))   # Validation data for monitoring during training
    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate( test_images_prep, test_labels_cat)

    # Print test results for reference
    print("Test Loss:", test_loss)  # Final loss on the test dataset
    print("Test Accuracy:", test_accuracy)    # Final accuracy on the test dataset
    
    # Save training and test results
    save_epoch_results(history, test_loss, test_accuracy, "ResNet")

    # Plot accuracy and loss graphs for training and validation
    plt.figure(figsize=(12, 4))  # Set the figure size for clarity

    # Plot accuracy over epochs
    plt.subplot(1, 2, 1)  # Create the first subplot (accuracy)
    plt.plot(history.history['accuracy'], label='Train Accuracy')  # Training accuracy
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy') # Validation accuracy
    plt.title('Accuracy over Epochs')  # Add a title to the accuracy plot
    plt.xlabel('Epoch')  # Label for the x-axis
    plt.ylabel('Accuracy')  # Label for the y-axis
    plt.legend()  # Add a legend to distinguish between train and validation accuracy

    # Plot loss over epochs
    plt.subplot(1, 2, 2)  # Create the second subplot (loss)
    plt.plot(history.history['loss'], label='Train Loss')   # Training loss
    plt.plot(history.history['val_loss'], label='Validation Loss')  # Validation loss
    plt.title('Loss over Epochs')  # Add a title to the loss plot
    plt.xlabel('Epoch')  # Label for the x-axis
    plt.ylabel('Loss')   # Label for the y-axis
    plt.legend()  # Add a legend to distinguish between train and validation loss
    
    # Add a main title for the entire figure
    plt.suptitle("The result of resnet50",fontsize=16)   # Emphasize the plot with a main title
    
    # Save the plot to a file
    plt.savefig("./images/task_B/Training and testing resnet50.png")   # Save the figure as an image
    return history  # Return the training history object


# normalise the data to range 0-1
def NormalisationB(train_images,val_images,test_images):
    train_images_normalized = train_images.astype('float32') / 255.0 # as the image is 0-225
    val_images_normalized = val_images.astype('float32') / 255.0
    test_images_normalized = test_images.astype('float32') / 255.0
    #check the output range of mornalise value
    print("range of the data after normlisation：")
    print(f"training set：{train_images_normalized.min(), train_images_normalized.max()}")
    print(f"validation set：{val_images_normalized.min(), val_images_normalized.max()}")
    print(f"testing set：{test_images_normalized.min(), test_images_normalized.max()}")
    
    # Return the normalized datasets
    return train_images_normalized,val_images_normalized,test_images_normalized

# Plot a sample of images, ensuring each label is represented
def plot_sample(images, labels):
    #make sure each lable as one image
    unique_labels = np.unique(labels)
    # Create a figure to display the samples
    plt.figure(figsize=(15, 15))
    #random select and plot
    for i, label in enumerate(unique_labels):
        idxs = np.where(labels == label)[0]  # Find indices of images with the current label
        random_idx = np.random.choice(idxs)  # Randomly select one index

        # Add the selected image to the subplot
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[random_idx]) # Display the image
        plt.title(f'Label: {label}') # Add the label as the title
        plt.axis('off')  # Remove axis for a cleaner look
    
    # Add a main title for the entire figure
    plt.suptitle('Random Sample for each lable')
    # Save the figure to a file
    plt.savefig("./images/task_B/SampleB.png")


# function of loading data
def Dataread(file_path):
    # load data from file
    dataset = np.load(file_path)
    #check the categries of the dataset
    dataset.files
    return dataset
# function of separate data into three different dataset
def category_Data(dataset):
    #load each category as a variable
    #train data 
    train_images = dataset['train_images']   # Training images
    train_labels = dataset['train_labels']  # Training labels
    #Validation data
    val_images = dataset['val_images']   # Validation images
    val_labels = dataset['val_labels']   # Validation labels
    #test data
    test_images = dataset['test_images']   # Test images
    test_labels = dataset['test_labels']  # Test labels

    # print the varibles
    #train
    print("Train Images Shape:", train_images.shape)
    print("Train Labels Shape:", train_labels.shape)
    print("\n")
    #Validation
    print("Validation Image Shape:", val_images.shape)
    print("Validation Labels Shape:", val_labels.shape)
    print("\n")
    #test
    print("Test Images Shape:", test_images.shape)
    print("Test Labels Shape:", test_labels.shape)
    print("\n")

    # close the file
    dataset.close()
    # Return the separated datasets
    return  train_images,train_labels,val_images,val_labels,test_images,test_labels



# Prepare the dataset by loading, normalizing, and sampling
def dataPrepare():
    
    # Define the file path to the dataset
    file_path = './Datasets/bloodmnist.npz'
    
    # Step 1: Load the dataset
    dataset=Dataread(file_path)

    # Step 2: Separate the dataset into training, validation, and test subsets
    train_images,train_labels,val_images,val_labels,test_images,test_labels=category_Data(dataset)

    # Step 3: Plot a random sample of training images to visually inspect the dataset
    plot_sample(train_images, train_labels)

    # Step 4: Normalize the training, validation, and test datasets
    train_images_normalized,val_images_normalized,test_images_normalized=NormalisationB(train_images,val_images,test_images)
    # resize the image from normalised data resize from 28-28 to 224-224
    # Return both raw and normalized datasets
    return train_images,train_labels,val_images,val_labels,test_images,test_labels,train_images_normalized,val_images_normalized,test_images_normalized

# Plot training progress for a given model
def plot_training_progress(history, model_name, output_path):
    """
    Plot training progress (accuracy and loss) for a given model.
    """
    
    # Extract the range of epochs
    epochs = range(1, len(history.history['accuracy']) + 1)

    # Create a figure for the plots
    plt.figure(figsize=(12, 6))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)  # First subplot (1 row, 2 columns, position 1)
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')  # Plot training accuracy
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')  # Plot validation accuracy
    plt.title(f"{model_name} Accuracy Over Epochs")   # Add title
    plt.xlabel("Epochs")  # Label x-axis
    plt.ylabel("Accuracy")  # Label y-axis
    plt.legend()  # Add legend to differentiate lines

    # Plot Loss
    plt.subplot(1, 2, 2)   # Second subplot (1 row, 2 columns, position 2)
    plt.plot(epochs, history.history['loss'], label='Training Loss')  # Plot training loss
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')  # Plot validation loss
    plt.title(f"{model_name} Loss Over Epochs")  # Add title
    plt.xlabel("Epochs")  # Label x-axis
    plt.ylabel("Loss")  # Label y-axis
    plt.legend()  # Add legend to differentiate lines

    # Adjust layout to avoid overlapping and save the figure
    plt.tight_layout()
    plt.savefig(output_path)  # Save the plot as a file
    print(f"{model_name} training progress saved to {output_path}")
    plt.close()  # Close the plot to free memory


def save_results_to_csv(history, csv_file):
    """
    Save training history results to a CSV file.
    
    Parameters:
        history (History): Training history object from Keras.
        csv_file (str): Path to the CSV file to save results.
    """
    # Convert the training history into a dictionary for structured data
    history_dict = {
        "epoch": list(range(1, len(history.history['accuracy']) + 1)),   # Epoch numbers
        "accuracy": history.history['accuracy'],  # Training accuracy for each epoch
        "val_accuracy": history.history['val_accuracy'],  # Validation accuracy for each epoch
        "loss": history.history['loss'],  # Training loss for each epoch
        "val_loss": history.history['val_loss']  # Validation loss for each epoch
    }
    
    # Convert the dictionary to a Pandas DataFrame for easy saving and manipulation
    df = pd.DataFrame(history_dict)
    
    # Save the DataFrame to the specified CSV file
    df.to_csv(csv_file, index=False)  # Do not include row indices in the CSV
    print(f"Results saved to {csv_file}")  # Notify that results were saved successfully
    print(df)   # Print the DataFrame for quick inspection



# **2. Main Execution**
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("./images/task_B", exist_ok=True)  # Ensure directory for images exists
    os.makedirs("./results/task_B_result", exist_ok=True)  # Ensure directory for results exists

    print("Preparing data...")
    # Step 1: Data Preparation
     # Load, separate, and normalize the dataset
    train_images, train_labels, val_images, val_labels, test_images, test_labels, \
    train_images_normalized, val_images_normalized, test_images_normalized = dataPrepare()

    print("Data preparation completed.")
    print(f"Train Images Shape: {train_images.shape}")  # Shape of training images
    print(f"Validation Images Shape: {val_images.shape}")   # Shape of validation images
    print(f"Test Images Shape: {test_images.shape}")   # Shape of test images

    # Step 2: Train Self-Designed CNN Model
    print("\nTraining Self-Designed CNN Model...")
    history_cnn = train_model(
        train_images_normalized,  # Normalized training images
        val_images_normalized,  # Normalized validation images
        test_images_normalized,  # Normalized test images
        train_labels,   # Training labels
        val_labels,   # Validation labels
        test_labels  # Test labels
    )
    print("Self-Designed CNN training completed.")
    
    # Plot and save the training progress for the Self-Designed CNN
    plot_training_progress(
        history_cnn,
        "Self-Designed CNN",   # Model name
        "./images/task_B/cnn_training_progress.png"   # Path to save the plot
    )

    # Step 3: Train ResNet50 Model
    print("\nTraining result of ResNet50 Model...")
    history_resnet = CNN_resnet(
        train_images_normalized,   # Normalized training images
        train_labels,   # Training labels
        val_images_normalized,   # Normalized validation images
        val_labels,   # Validation labels
        test_images_normalized,  # Normalized test images
        test_labels    # Test labels
    )
    print("ResNet50 training completed.")
    
    # Plot and save the training progress for the ResNet50 Model
    plot_training_progress(
        history_resnet,
        "ResNet50",  # Model name
        "./images/task_B/resnet_training_progress.png"  # Path to save the plot
    )

    print("\nTraining completed for both models.")
    print("Results and training progress saved in './images/task_B' and 'AMLS_24_25_SN21054037/results'.")
    


    
    

