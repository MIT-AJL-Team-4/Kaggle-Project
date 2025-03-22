# Equitable AI for Dermatology

## Contributors 
**Reasearch:**
- Maryam Choudhury 
- Jason Jiang
- Yuri Lee
- Katie Wilson
- Caroline Zouloumain

**Initial Creation of EfficientNetB0_ImageNet and ResNet50_ImageNet_SkinClassifier:**
- Katie Wilson
- Caroline Zouloumian

**Initial Creation of TransferLearning_Efficient**
- Maryam Choudhury

## Overview

In this project our team is addressing the  need for inclusive dermatology AI by experimenting with developing a model that "accurately classifies 21 skin conditions across a diverse range of skin tones" (Raghavan, 2025). This project is a part of the Break Through Tech and Algorithmic Justice League Kaggle competition.

## Model Architecture

Our solution leverages EfficientNetB0, a CNN pre-trained on ImageNet, a dataset we chose after learning about it's vast bredth of labeled skin condition images and thus great potential for training models. By employing transfer learning, we fine-tune EfficientNetB0 on the dermatology dataset provided on the AJL Kaggle competition board. 

## Data & Training

**Dataset:** We utilize the [AJL Kaggle dataset](https://www.kaggle.com/competitions/bttai-ajl-2025), which encompasses images representing a wide spectrum of skin tones and 21 distinct dermatological conditions.

> "The dataset is a subset of the FitzPatrick17k dataset, a labeled collection of about 17,000 images depicting a variety of serious (e.g., melanoma) and cosmetic (e.g., acne) dermatological conditions with a range of skin tones scored on the FitzPatrick skin tone scale (FST). About 4500 images are in this set, representing 21 skin conditions out of the 100+ in the full FitzPatrick set" (AJL, Equitable AI for Dermatology, 2025).

**Preprocessing:**

**Evaluation:** Model performance is measured using a weighted F1 score
