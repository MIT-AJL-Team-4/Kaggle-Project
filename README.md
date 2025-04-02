# Equitable AI for Dermatology

## Contributors 

| Name | GitHub Handle | Contributions |
|------|--------------|---------------|
| Maryam Choudhury | @maryam-choudhury | **Research**: Comparative research on CNN architectures (EfficientNet, ResNet, VGG)<br>**Model Building**: Creation of TrannsferLearning_ResNet50 |
| Katie Wilson | @kwilson44 | **Research, Model Building**: Creation of EfficientNetB0_ImageClassifier and ResNet |
| Caroline Zouloumian | @czouloumian | **Research, Model Building using ResNet** |
| Yuri Lee | @j-yuri-lee | **Research, Model Exploration**: Researched EfficientNet-B7 and explored training on a filtered Fitzpatrick dataset |
| Jason Jiang | GitHub Handle | **Research:** Pre-trained models |
| Marelyn Gonzalez | @mdgo23 | **Research:** Pre-trained models |

## Overview

We made this project as part of the Break Through Tech AI Spring Kaggle competition, in collaboration with the Algorithmic Justice League. The Algorithmic Justice League was founded in 2016 by Dr. Joy Buolamwini to raise awareness about bias in AI, especially with face recognition.

In this project, our team is addressing the need for inclusive dermatology AI by developing a model that "accurately classifies 21 skin conditions across a diverse range of skin tones" (Raghavan, 2025). As image recognition plays an increasing role in dermatological diagnosis, darker skin tones remain underrepresented. The goal of this project is to improve dermatology diagnosis for all.

## Model Architecture

Our solutions leverage:
- **EfficientNetB0**, a CNN pre-trained on ImageNet. However, this model underperformed significantly, achieving an F1 score of **0.04**.
- **ResNet50**, a pre-trained model developed by Microsoft in 2015. This CNN is used for image classification and was fine-tuned on the dermatology dataset, achieving an F1 score of **38**.
- **TransferLearning_ResNet**, our best-performing model, which achieved an F1 score of **52**.

## Data & Training

**Dataset:** We utilize the [AJL Kaggle dataset](https://www.kaggle.com/competitions/bttai-ajl-2025), which encompasses images representing a wide spectrum of skin tones and 21 distinct dermatological conditions.

> "The dataset is a subset of the FitzPatrick17k dataset, a labeled collection of about 17,000 images depicting a variety of serious (e.g., melanoma) and cosmetic (e.g., acne) dermatological conditions with a range of skin tones scored on the FitzPatrick skin tone scale (FST). About 4500 images are in this set, representing 21 skin conditions out of the 100+ in the full FitzPatrick set" (AJL, Equitable AI for Dermatology, 2025).

**Preprocessing:**
We used data augmentation to expand our dataset and improve model robustness.

**Evaluation:**
Model performance is measured using a weighted F1 score.

## Impact Narrative

A key motivation for this project is addressing bias in AI-driven dermatology tools. Many existing models struggle with classifying skin conditions on darker skin tones due to imbalanced training datasets. By fine-tuning models on a more representative dataset, we aim to make dermatological AI more equitable.

## Next Steps & Future Improvements

- **Dataset Expansion:** Incorporating additional high-quality dermatology datasets to improve generalization.
- **Model Optimization:** Further hyperparameter tuning and architectural changes to enhance F1 scores.
- **Bias Mitigation:** Exploring techniques such as re-weighting loss functions to balance performance across skin tones.
- **Clinical Validation:** Engaging with dermatologists to assess real-world applicability of the model.
