import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

# Define paths
train_dir = "/Users/katiewilson/Desktop/bttai-ajl-2025/train/train"
test_dir = "/Users/katiewilson/Desktop/bttai-ajl-2025/test/test"

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2  # Splitting train data for validation
)

# Load Data from Directories
batch_size = 32
img_size = (224, 224)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'
)
val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

# Create the ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Compute Class Weights
class_labels = np.array(train_generator.classes)
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(class_labels), y=class_labels
)
class_weights_dict = dict(enumerate(class_weights))

# Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=32,  # Adjust as needed
    class_weight=class_weights_dict
)

# Unfreeze last 10 layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Fine-tune the model
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights_dict
)

# Prepare Test Data for Predictions
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
test_generator = test_datagen.flow_from_directory(
    directory=os.path.dirname(test_dir),  # Parent directory
    target_size=img_size,
    batch_size=batch_size,
    shuffle=False,
    class_mode=None  # No labels in test set
)

# Make Predictions
test_preds = model.predict(test_generator, steps=int(np.ceil(len(test_generator.filenames) / batch_size)))
test_labels = np.argmax(test_preds, axis=1)

# Get the label mapping (index to class name)
label_map = {v: k for k, v in train_generator.class_indices.items()}

# Map numerical predictions to class names
predicted_class_labels = [label_map[label_idx] for label_idx in test_labels]

# Evaluation on Validation Set
val_generator_for_metrics = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

y_true = val_generator_for_metrics.classes
val_preds = model.predict(val_generator_for_metrics, steps=int(np.ceil(len(val_generator_for_metrics.filenames) / batch_size)))
y_pred = np.argmax(val_preds, axis=1)

# Calculate F1 Scores
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(train_generator.class_indices.keys())))

f1_macro = f1_score(y_true, y_pred, average='macro')
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print("\nF1 Scores:")
print(f"F1 Macro: {f1_macro:.4f}")
print(f"F1 Micro: {f1_micro:.4f}")
print(f"F1 Weighted: {f1_weighted:.4f}")

# Convert Predictions to DataFrame
test_filenames = [os.path.basename(f) for f in test_generator.filenames]

# Create the submission DataFrame
submission_df = pd.DataFrame({
    "md5hash": test_filenames,
    "label": predicted_class_labels
})

submission_filename = "submission.csv"
submission_df.to_csv(submission_filename, index=False)
print(f"{submission_filename} saved successfully!")
