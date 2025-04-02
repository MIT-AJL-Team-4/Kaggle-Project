# Equitable AI for Dermatology

## Contributors

| Name | GitHub Handle | Contributions |
|------|--------------|---------------|
| Maryam Choudhury | @maryam-choudhury | **Research**: Comparative research on CNN architectures (EfficientNet, ResNet, VGG) | **Model Building**: Creation of ResNet50_ImageNet_SkinClassifier |
| Katie Wilson | @kwilson44 | **Research, Model Building**: Creation of EfficientNetB0_ImageClassifier and ResNet |
| Caroline Zouloumian | @czouloumian | **Research, Model Building using ResNet** |
| Yuri Lee | @j-yuri-lee | **Research, Model Exploration**: Researched EfficientNet-B7 and explored training on a filtered Fitzpatrick dataset |
| Jason Jiang | GitHub Handle | **Research pre-trained models** |
| Marelyn Gonzalez | @mdgo23 | **Researched pre-trained models** |

---

**Initial Creation of EfficientNetB0_ImageNet and ResNet50_ImageNet_SkinClassifier:**
- Katie Wilson
- Caroline Zouloumian

**Initial Creation of TransferLearning_Efficient:**
- Maryam Choudhury

## Overview

This project was developed as part of the Break Through Tech AI Spring Kaggle competition, in collaboration with the Algorithmic Justice League. The Algorithmic Justice League was founded in 2016 by Dr. Joy Buolamwini to raise awareness about bias in AI, particularly in facial recognition systems.

Our team aims to address the need for inclusive dermatology AI by developing a model that "accurately classifies 21 skin conditions across a diverse range of skin tones" (Raghavan, 2025). As AI-powered image recognition increasingly assists dermatological diagnosis, darker skin tones remain underrepresented in available datasets. Our goal is to improve dermatology AI for all.

## Model Architecture

Our approach leverages the following models:

- **EfficientNetB0**: A CNN pre-trained on ImageNet. However, this model underperformed significantly, achieving an F1 score of **0.04**.
- **ResNet50**: A pre-trained model developed by Microsoft in 2015 for image classification. Fine-tuned on our dermatology dataset, it achieved an F1 score of **38**.
- **TransferLearning_Efficient**: Our best-performing model, which achieved an F1 score of **52**.

## Data & Training

**Dataset:** We utilize the [AJL Kaggle dataset](https://www.kaggle.com/competitions/bttai-ajl-2025), which includes images spanning a broad range of skin tones and 21 dermatological conditions.

> "The dataset is a subset of the FitzPatrick17k dataset, a labeled collection of about 17,000 images depicting various serious (e.g., melanoma) and cosmetic (e.g., acne) dermatological conditions across different skin tones, scored on the FitzPatrick skin tone scale (FST). The subset used for this challenge contains approximately 4,500 images representing 21 conditions" (AJL, Equitable AI for Dermatology, 2025).

**Preprocessing:**
We applied data augmentation techniques, including rotation, shifting, zooming, brightness adjustment, and horizontal flipping, to improve generalization and address class imbalances.

**Evaluation:**
Model performance is measured using a weighted **F1-score**, which aligns with Kaggle's metric for this competition.

## References

- [Deep Learning in Dermatology: A Systematic Review of Current Approaches, Outcomes, and Limitations - ScienceDirect](https://www.nature.com/articles/s41467-024-50043-3)
- [SkinGPT-4 GitHub Repository](https://github.com/JoshuaChou2018/SkinGPT-4)
- [NVIDIA Blog: Building Real-Time Dermatology Classification](https://developer.nvidia.com/blog/building-real-time-dermatology-classification-with-nvidia-clara-agx/)
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [AJL Starter Notebook](https://www.kaggle.com/competitions/bttai-ajl-2025)

