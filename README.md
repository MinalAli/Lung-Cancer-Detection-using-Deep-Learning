# Lung Cancer Classification using Deep Learning

This project implements a deep learning model for classifying different types of lung cancer using medical imaging data. The model utilizes transfer learning with the Xception architecture to classify lung CT scans into four categories: normal, adenocarcinoma, large cell carcinoma, and squamous cell carcinoma.

## Table of Contents
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Organization](#dataset-organization)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Results](#results)

## Dependencies

```bash
# Core Libraries
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
keras
```

## Project Structure

```
.
├── Lung_Cancer_Prediction.py
├── README.md
├── dataset
│ ├── train
│ │ ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
│ │ ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
│ │ ├── normal
│ │ └── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa
│ ├── test
│ │ ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
│ │ ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
│ │ ├── normal
│ │ └── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa
│ └── valid
│ ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
│ ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
│ ├── normal
│ └── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa
└── best_model.keras
```

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn keras
```

## Dataset Organization

The dataset should be organized in the following structure:
- Training data: `./dataset/train/`
- Testing data: `./dataset/test/`
- Validation data: `./dataset/valid/`

Each directory contains subdirectories for the four classes:
- Normal scans
- Adenocarcinoma
- Large cell carcinoma
- Squamous cell carcinoma

## Model Architecture

The model uses transfer learning with the following architecture:

```python
# Base Model: Xception (pre-trained on ImageNet)
pretrained_model = tf.keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    input_shape=[350, 350, 3]
)
pretrained_model.trainable = False

# Additional layers for classification
model = Sequential([
    pretrained_model,
    GlobalAveragePooling2D(),
    Dense(4, activation='softmax')
])
```

## Training Process

1. Image Preprocessing:
   - Images are resized to 350x350 pixels
   - Pixel values are normalized to [0,1]
   - Data augmentation includes horizontal flipping for training data

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)
```

2. Training Configuration:
   - Batch size: 8
   - Learning rate reduction on plateau
   - Early stopping to prevent overfitting
   - Model checkpointing to save the best model

```python
# Training callbacks
learning_rate_reduction = ReduceLROnPlateau(
    monitor='loss',
    patience=5,
    factor=0.5,
    min_lr=0.000001
)
early_stops = EarlyStopping(
    monitor='loss',
    patience=6,
    mode='auto'
)
```

## Usage

To classify a new image:

```python
def load_and_preprocess_image(img_path, target_size=(350, 350)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Load and predict
img = load_and_preprocess_image('path_to_image.jpg')
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])
```

## Model Performance

The model's performance can be monitored through:
- Training and validation accuracy curves
- Training and validation loss curves
- Final training accuracy
- Final validation accuracy

The trained model is saved as 'trained_lung_cancer_model.keras' and can be loaded for inference.

## Results

The video shows:
- Model predictions on various test images
- Visualization of classification results
- Performance metrics and accuracy scores

[Watch the video](https://youtu.be/xutlsjAK_io)

The video demonstrates the model's ability to accurately classify different types of lung cancer from CT scan images, showcasing its potential as a diagnostic aid tool.

## Contributing

Feel free to submit issues and enhancement requests!


