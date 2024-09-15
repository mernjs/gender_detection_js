# Gender Detection

## Overview
This project demonstrates a gender detection model using TensorFlow and Keras. The model is trained on a dataset of images classified by gender (male/female).

## Folder Structure
- `dataset/`: Contains the training and validation images.
- `src/`: Contains the code for training, evaluating, and predicting.
- `requirements.txt`: Lists the required Python packages.
- `README.md`: This file.

### Installation

- Create and activate a virtual environment.
- Install dependencies from `requirements.txt`.

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
    ```bash
    source venv/bin/activate
    ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

2. **Train the Model**:
   ```bash
   python src/train_model.py
   ```

3. **Evaluate the Model**:
   ```bash
   python src/evaluate_model.py
   ```

4. **Predict Gender from an Image**:
   ```bash
   python src/predict_image.py --image path_to_image.jpg
   ```