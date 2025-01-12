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
        # Use subprocess to run the script and capture its output
        result = subprocess.run(
            ["python", script_name],    # Command to run the Python script
            capture_output=True,        # Capture both stdout and stderr
            text=True,                  # Decode the output to a string format
            check=True
        )
        return result.stdout   # Return the standard output of the script
    except subprocess.CalledProcessError as e:
        # Handle errors if the script fails to execute
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
    # Use a regular expression to find "Test Accuracy: <value>" in the script output
    match = re.search(r"Test Accuracy: ([0-9.]+)", output)
    if match:
        return float(match.group(1)) # Return the extracted accuracy as a float
    else:
        print("Could not find test accuracy in the output.")
        return None  # Return None if accuracy is not found



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
    
    # Create a bar chart to display accuracies for each method
    plt.figure(figsize=(10, 6))
    plt.bar(methods, accuracies, color=['blue', 'green', 'orange', 'purple'])
    plt.title("Test Accuracies of CNN (Custom), CNN (with augmented data), SVM, and K-NN")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1) # Set y-axis range to [0, 1] for clarity
    
    # Annotate bars with accuracy values
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f"{acc:.4f}", ha='center', fontsize=12)
        
    # Save the plot to the specified output path
    plt.savefig(output_path)
    print(f"Results graph saved to {output_path}")
    plt.close()



# **Main Execution**
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("./results/task_A_result", exist_ok=True)  # Create results directory
    
    # Initialize a dictionary to store the test accuracies for each method
    results = {}

    # Step 1: Run CNN (Custom) script and record results
    print("Running CNN (Custom) training...")
    os.system("python ./A/main_A_CNN.py")  # Run the custom CNN script
    cnn_custom_output_file = "./results/task_A_result/results_table_original.csv"
    if os.path.exists(cnn_custom_output_file):
        # Read the final accuracy from the output CSV file
        import pandas as pd
        cnn_custom_df = pd.read_csv(cnn_custom_output_file)
        results["CNN (Custom)"] = cnn_custom_df.iloc[-1]["accuracy"]
        print(f"CNN (Custom) Test Accuracy: {results['CNN (Custom)']}")

    # Step 2: Run CNN (augmented) script and record results
    print("Running augmented training...")
    augmented_output_file = "./results/task_A_result/results_table_augmented.csv"
    if os.path.exists(augmented_output_file):
        # Read the final accuracy from the augmented data CSV file
        cnn_augmented_df = pd.read_csv(augmented_output_file)
        results["CNN_augmented"] = cnn_augmented_df.iloc[-1]["accuracy"]
        print(f"CNN (augmented) Test Accuracy: {results['CNN_augmented']}")

    # Step 3: Run SVM and K-NN scripts and record their results
    scripts = {
        "SVM": "./A/main_A_SVM.py", # Path to the SVM script
        "K-NN": "./A/main_A_K_NN.py", # Path to the K-NN script
    }
    # Execute each script and capture its output
    for method, script in scripts.items():
        print(f"Running {method} script: {script}")
        output = run_script(script)  # Run the script and get its output
        if output:
            accuracy = extract_accuracy(output)  # Extract accuracy from the script output
            if accuracy is not None:
                results[method] = accuracy
                print(f"{method} Test Accuracy: {accuracy}")
            else:
                print(f"Failed to extract accuracy for {method}.")
        else:
            print(f"Failed to run script {script}.")

    # Step 4: Plot results if all scripts were successful
    if results:
        plot_results(results, "./results/task_A_result/task_A_test_accuracies.png")



