# Medical Image Classification Project

This repository contains Python scripts for training and evaluating machine learning and deep learning models for medical image classification tasks. The project is designed to work on **Windows systems** and requires specific libraries, datasets, and configurations to run successfully.

## Project Structure

```
├── A/                          # Task A: Binary Classification Scripts
│   ├── main_A.py               # Controller script for Task A
│   ├── main_A_CNN.py           # Custom CNN implementation
│   ├── main_A_SVM.py           # SVM implementation
│   ├── main_A_K_NN.py          # K-NN implementation
├── B/                          # Task B: Multi-Class Classification Scripts
│   ├── main_B.py               # Controller script for Task B
│   ├── main_B_CNN.py           # Custom CNN and ResNet50 implementation
│   ├── main_B_SVM.py           # SVM implementation
│   ├── main_B_K_NN.py          # K-NN implementation
├── Datasets/                   # Input datasets for Task A and Task B
├── images/                     # Folder for saved plots and visualizations
├── results/                    # Folder for combined results (CSV files, performance metrics)
├── main.py                     # Main entry point for the project
├── README.md                   # Project description and instructions
└── requirements.txt            # Python dependencies
```



---

## Prerequisites

### Operating System
- **Windows 10 or higher**

### Required Libraries
The code depends on several Python libraries listed in `requirements.txt`. Ensure you have Python 3.8 or higher installed.

---

## Setup Instructions

1. **Download the code to create a Virsual environment**:
   If your computer is a Windows system, Run the following command in the terminal one by one to create an virtual environment:

    This is for creating an empty virtual environment:
      ```bash
      python -m venv .venv     
      ```
    This is for activate the virtual environment:
      ```bash
      .\.venv\Scripts\activate      
      ```
After activate the virtual environment, you can install the following libraries as the instruction shown below:

3. **Install Required Libraries**:
   Run the following command to install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the Dataset**:
   - Place the required datasets in the `Datasets` folder. 
   - Ensure the dataset files match the expected format (.npz).

---

## Running the Code

### Command to Run
To execute the main program, use the following command from the terminal:
```bash
python main.py
```

### What Happens Next
- During the training process, the following folders will be created automatically:
  1. `results`: Stores training outcomes, including CSV files with metrics such as accuracy and loss.
  2. `images`: Contains plots of training and validation metrics.

---

## Outputs

1. **Results**:
   - Training and validation metrics (accuracy, loss) are saved in the `results` folder as CSV files.

2. **Visualizations**:
   - Plots of training and validation performance are saved in the `images` folder.

---



---

## Notes
- Ensure that the `Datasets` folder contains the correct dataset files before running the code.
- If any issues arise during the installation or execution, please raise an issue in the repository.

---
