import os
import subprocess
import re
import matplotlib.pyplot as plt


# **1. Helper Function to Run Scripts**
def run_script(script_name):
    """
    Run a Python script and capture its output.
    
    Parameters:
        script_name (str): The name of the Python script to run.
    
    Returns:
        str: The stdout output of the script.
    """
    try:
        # Run the script using subprocess and capture the output
        result = subprocess.run(
            ["python", script_name], capture_output=True, text=True, check=True
        )
        return result.stdout   # Return the standard output
    except subprocess.CalledProcessError as e:
        
        # Print error details if the script fails to execute
        print(f"Error running {script_name}: {e.stderr}")
        return None

# **2. Extract Accuracy from Script Output**
def extract_accuracy(output):
    """
    Extract the test accuracy from the output of a script.
    
    Parameters:
        output (str): The stdout output of a script.
    
    Returns:
        float: The test accuracy as a decimal value.
    """
    
    # Use regex to search for "Test Accuracy: <value>" in the script output
    match = re.search(r"Test Accuracy: ([0-9.]+)", output)
    if match:
        return float(match.group(1))  # Return the extracted accuracy as a float
    else:
        print("Could not find test accuracy in the output.")
        return None   # Return None if accuracy is not found

# **3. Plot Results**
def plot_results(results, output_path):
    """
    Plot a bar graph of the test accuracies and save it to a file.
    
    Parameters:
        results (dict): A dictionary with model names as keys and accuracies as values.
        output_path (str): Path to save the plot.
    """
    
    # Extract model names and their corresponding accuracies
    methods = list(results.keys())
    accuracies = list(results.values())
    
    # Create a bar chart to display the accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(methods, accuracies, color=['blue', 'green', 'orange', 'purple'])
    plt.title("Test Accuracies of CNN (Custom), CNN (ResNet50), SVM, and K-NN")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)  # Set y-axis range to [0, 1] for better visualization
    for i, acc in enumerate(accuracies):
        
        # Annotate each bar with the accuracy value
        plt.text(i, acc + 0.02, f"{acc:.4f}", ha='center', fontsize=12)
        
    # Save the plot to the specified path
    plt.savefig(output_path)
    print(f"Results graph saved to {output_path}")
    plt.close()  # Close the plot to free memory


# **Main Execution**
if __name__ == "__main__":
    # Step 1: Create results directory if it doesn't exist
    os.makedirs("./results/task_B_result", exist_ok=True)  # Create results directory
    
    # Initialize a dictionary to store the test accuracies for each method
    results = {}

   # Step 2: Run CNN (Custom) training and record results
    print("Running CNN (Custom) training...")
    os.system("python ./B/main_B_CNN.py")  # Run the script for the custom CNN
    cnn_custom_output_file = "./results/task_B_result/cnn_custom_epoch_results.csv"
    if os.path.exists(cnn_custom_output_file):
        # Read the test accuracy from the output CSV file
        import pandas as pd
        cnn_custom_df = pd.read_csv(cnn_custom_output_file)
        results["CNN_train_model"] = cnn_custom_df.iloc[-1]["test_accuracy"]
        print(f"CNN (Custom) Test Accuracy: {results['CNN_train_model']}")

    # Step 3: Run CNN (ResNet50) training and record results
    print("Running ResNet50 training...")
    resnet_output_file = "./results/task_B_result/resnet_epoch_results.csv"
    # Read the test accuracy from the output CSV file
    if os.path.exists(resnet_output_file):
        # Read last epoch's accuracy
        resnet_df = pd.read_csv(resnet_output_file)
        results["CNN_resnet"] = resnet_df.iloc[-1]["test_accuracy"]
        print(f"CNN (ResNet50) Test Accuracy: {results['CNN_resnet']}")

    # Step 4: Run SVM and K-NN scripts and record their results
    scripts = {
        "SVM": "./B/main_B_SVM.py",   # Script for SVM
        "K-NN": "./B/main_B_K_NN.py",  # Script for K-NN
    }

    for method, script in scripts.items():
        print(f"Running {method} script: {script}")  # Run the script and capture its output
        output = run_script(script)
        if output:
            accuracy = extract_accuracy(output)   # Extract accuracy from the script output
            if accuracy is not None:
                results[method] = accuracy
                print(f"{method} Test Accuracy: {accuracy}")
            else:
                print(f"Failed to extract accuracy for {method}.")
        else:
            print(f"Failed to run script {script}.")

   # Step 5: Plot results if all scripts were successful
    if results:
        plot_results(results, "./results/task_B_result/task_B_test_accuracies.png")


