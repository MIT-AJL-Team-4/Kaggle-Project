# Kaggle Dermatology Classification Challenge

## Overview
This project trains a deep learning model using **ResNet50** to classify skin conditions from images. The trained model generates predictions that need to be formatted correctly for submission to Kaggle.

## Model Details
The model is built using **ResNet50**, a deep convolutional neural network pretrained on ImageNet. The base model is initially frozen, and a custom classifier with **GlobalAveragePooling2D**, **Dense**, and **Dropout** layers is added. The model is trained using **sparse categorical cross-entropy loss** and optimized with **Adam**. Fine-tuning is performed by unfreezing the last 10 layers of ResNet50 to improve performance.

Class imbalances are addressed using **class weights**, and **data augmentation** (rotation, shifting, zooming, brightness adjustment, and horizontal flipping) is applied to enhance generalization. Model evaluation is conducted using **F1-score** to align with Kaggle's weighted metric.

## Files
- `restnet.py`: The main script that trains the model, evaluates performance, and generates `submission.csv`.
- `jpg_remover.py`: A script to process `submission.csv` and remove `.jpg` extensions from the `md5hash` column to make it Kaggle submission-ready.
- `submission.csv`: The raw output from `restnet.py` (requires processing before submission).
- `submission_fixed.csv`: The final, correctly formatted file after running `jpg_remover.py`.

## Setup
### Requirements
Ensure you have the following installed:
```sh
pip install tensorflow numpy pandas scikit-learn matplotlib
```

## Running the Model
Train the model and generate predictions by running:
```sh
python restnet.py
```
This will output `submission.csv`.

## Formatting the Submission File
Before submitting to Kaggle, ensure that `.jpg` extensions are removed from the `md5hash` column by running:
```sh
python jpg_remover.py
```
This will generate `submission_fixed.csv`, which is ready for submission.

## Kaggle Submission Format
The final CSV should be formatted as:
```csv
md5hash,label
16d1e6b4143c88cb158a50ea8bc3a595,acne-vulgaris
aceebbdcfd419fa960ebe3933d721550,folliculitis
85bfb7325d93bac71fcbc08ae0a9ba23,dermatomyositis
```

## Notes
- Ensure your training and test directories are correctly set in `restnet.py`.
- If Kaggle still rejects the submission, check that `jpg_remover.py` properly processed the file.

