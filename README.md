# Equitable AI for Dermatology

## Overview
This project was developed as part of the **Break Through Tech AI Spring Kaggle competition** in collaboration with the **Algorithmic Justice League (AJL)**. Our goal was to address bias in AI-driven dermatology by developing a model that accurately classifies **21 skin conditions across a diverse range of skin tones**.

Darker skin tones are often underrepresented in dermatological datasets, which can lead to misdiagnosis and inequitable healthcare outcomes. By training deep learning models on a **diverse dataset**, we aim to improve dermatology diagnosis for all skin types.

## Contributors
| Name               | GitHub Handle    | Contributions  |
|-------------------|----------------|---------------------------------------------|
| **Maryam Choudhury** | @maryam-choudhury | Research on CNN architectures (EfficientNet, ResNet, VGG); Model building: ResNet50_ImageNet_SkinClassifier |
| **Katie Wilson** | @kwilson44 | Model building: EfficientNetB0_ImageClassifier, ResNet; Research |
| **Caroline Zouloumian** | @czouloumian | Model building: ResNet; Research |
| **Yuri Lee** | @j-yuri-lee | Model exploration: EfficientNet-B7; Filtered Fitzpatrick dataset research |
| **Jason Jiang** | GitHub Handle | Research |
| **Marelyn Gonzalez** | @mdgo23 | Researched pre-trained models |

## Model Architecture
We experimented with multiple deep learning models:
- **EfficientNetB0** (Pre-trained on ImageNet, transfer learning applied) - F1 Score: **0.04**
- **ResNet50** (Pre-trained on ImageNet, fine-tuned) - F1 Score: **38**
- **TransferLearning_Efficient (final model selected)** - F1 Score: **52**

### Final Model: ResNet50
The selected model, **ResNet50**, was fine-tuned with:
- **GlobalAveragePooling2D** and custom classification layers
- **Sparse categorical cross-entropy loss** with **Adam optimizer**
- **Data augmentation**: rotation, shifting, zooming, brightness adjustment, horizontal flipping
- **Class weighting** to mitigate dataset imbalances

## Data & Training
### Dataset
We used the **[AJL Kaggle dataset](https://www.kaggle.com/competitions/bttai-ajl-2025)**, which includes:
- **4,500 images** covering **21 dermatological conditions**
- A **subset of the FitzPatrick17k dataset**, which labels images based on skin tone diversity

### Preprocessing
- **Data Augmentation**: Increased training diversity
- **Balanced Training**: Adjusted for class imbalances
- **Normalization**: Rescaled images to enhance model performance

### Evaluation
- **Primary Metric**: **Weighted F1 Score** (aligned with Kaggle evaluation criteria)
- **Performance:**
  - **EfficientNetB0**: **F1 Score = 0.04**
  - **ResNet50**: **F1 Score = 38**
  - **Final Model (TransferLearning_Efficient)**: **F1 Score = 52**

## Files
- `resnet.py`: Trains the model, evaluates performance, and generates `submission.csv`
- `jpg_remover.py`: Processes `submission.csv` to remove `.jpg` extensions for Kaggle submission
- `submission.csv`: Raw model predictions
- `submission_fixed.csv`: Final formatted predictions ready for Kaggle

## Setup & Execution
### Install Dependencies
```sh
pip install tensorflow numpy pandas scikit-learn matplotlib
```

### Train the Model
```sh
python resnet.py
```
This generates `submission.csv`.

### Format the Submission File
Ensure `.jpg` extensions are removed before submission:
```sh
python jpg_remover.py
```
This generates `submission_fixed.csv`.

## Kaggle Submission Format
The final CSV should be formatted as:
```csv
md5hash,label
16d1e6b4143c88cb158a50ea8bc3a595,acne-vulgaris
aceebbdcfd419fa960ebe3933d721550,folliculitis
85bfb7325d93bac71fcbc08ae0a9ba23,dermatomyositis
```

## Impact Narrative
We prioritized **model fairness** by:
- **Training on diverse skin tones** to mitigate bias in dermatological diagnosis
- **Using class weights** to counteract dataset imbalances
- **Evaluating with an inclusive metric (weighted F1 score)** to ensure generalizability

## Next Steps & Future Improvements
- **Improve dataset diversity**: Expand training data to better represent all skin tones
- **Experiment with advanced architectures**: Test hybrid models (e.g., Vision Transformers + CNNs)
- **Optimize hyperparameters**: Further fine-tune learning rates, batch sizes, and regularization

## References
- [Deep Learning in Dermatology: A Systematic Review](https://www.nature.com/articles/s41467-024-50043-3)
- [SkinGPT-4 GitHub Repository](https://github.com/JoshuaChou2018/SkinGPT-4)
- [NVIDIA Dermatology Classification](https://developer.nvidia.com/blog/building-real-time-dermatology-classification-with-nvidia-clara-agx/)
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- **AJL Starter Notebook** (AJL Kaggle Competition Resources)

---
This project was made possible by **Break Through Tech AI** and the **Algorithmic Justice League**.

