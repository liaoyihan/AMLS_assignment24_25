# Medical Image Classification Project

This repository contains Python scripts for training and evaluating machine learning and deep learning models for medical image classification tasks. The project is designed to work on **Windows systems** and requires specific libraries, datasets, and configurations to run successfully.

---

## Prerequisites

### Operating System
- **Windows 10 or higher**

### Required Libraries
The code depends on several Python libraries listed in `requirements.txt`. Ensure you have Python 3.8 or higher installed.

---

## Setup Instructions

1. **Download the code create a Virsual environment**:
  

2. **Install Required Libraries**:
   Run the following command to install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add the Dataset**:
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

## Example Workflow

### Terminal Commands
1. Download the code, create a Virsual environemnt

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add datasets to the `Datasets` folder.

4. Run the code:
   ```bash
   python main.py
   ```

---

## Notes
- Ensure that the `Datasets` folder contains the correct dataset files before running the code.
- If any issues arise during the installation or execution, please raise an issue in the repository.

---
