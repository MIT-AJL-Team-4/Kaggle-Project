# Equitable AI for Dermatology

## Contributors 

| Maryam Choudhury | @maryam-choudhury | **Research**: Comparative research on CNN architectures (EfficientNet, ResNet, VGG) | **Model Building**: Creation of ResNet50_ImageNet_SkinClassifier |

| Katie Wilson | GitHub Handle | Research, Model Building: Creation of EfficientNetB0_ImageClassifier |

| Caroline Zouloumian  | @czouloumian | Research, Model Building using ResNet|

| Yuri Lee | j-yuri-lee | Research, Model Exploration: Researched EfficientNet-B7 and explored training on a filtered Fitzpatrick dataset |

| Jason Jiang | GitHub Handle | Research |

| Marelyn Gonzalez | @mdgo23 | Researched pre-trained models |

---


**Initial Creation of EfficientNetB0_ImageNet and ResNet50_ImageNet_SkinClassifier:**
- Katie Wilson
- Caroline Zouloumian

**Initial Creation of TransferLearning_Efficient**
- Maryam Choudhury


## Overview

We made this project as part of the Break Through Tech AI Spring Kaggle competition, in collaboration with the Algorithmic Justice League.The Algorithm Justice League was founded in 2016 by Dr. Joy Buolamwini to raise awareness about bias in AI, especially with face recognition.

In this project our team is addressing the  need for inclusive dermatology AI by experimenting with developing a model that "accurately classifies 21 skin conditions across a diverse range of skin tones" (Raghavan, 2025). Indeed, as image recognition is starting to play a bigger role in dermatological conditions recognition, darker skin tones remain underrepresented in the domain. The goal for this project is to improve dermatology diagnosis for all.


## Model Architecture

Our solutions leverages:
- EfficientNetB0, a CNN pre-trained on ImageNet, a dataset we chose after learning about it's vast bredth of labeled skin condition images and thus great potential for training models. By employing transfer learning, we fine-tune EfficientNetB0 on the dermatology dataset provided on the AJL Kaggle competition board.

- ResNet50 model, a pre-trained model developed by Microsoft in 2015. In particular, it is a CNN that is used for image classification. This model is also pre trained on ImageNet.


## Data & Training

**Dataset:** We utilize the [AJL Kaggle dataset](https://www.kaggle.com/competitions/bttai-ajl-2025), which encompasses images representing a wide spectrum of skin tones and 21 distinct dermatological conditions.

> "The dataset is a subset of the FitzPatrick17k dataset, a labeled collection of about 17,000 images depicting a variety of serious (e.g., melanoma) and cosmetic (e.g., acne) dermatological conditions with a range of skin tones scored on the FitzPatrick skin tone scale (FST). About 4500 images are in this set, representing 21 skin conditions out of the 100+ in the full FitzPatrick set" (AJL, Equitable AI for Dermatology, 2025).

**Preprocessing:**
We used data augmentation in order to train our model on more images.


**Evaluation:** Model performance is measured using a weighted F1 score
