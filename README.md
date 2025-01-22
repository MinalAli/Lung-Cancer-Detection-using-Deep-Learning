# Lung Cancer Classification Using Deep Learning

This repository contains code for classifying lung cancer images into four categories using a convolutional neural network (CNN) based on the Xception model, fine-tuned for this task. The project uses TensorFlow and Keras libraries for building, training, and evaluating the model.

## Features
- Preprocessing of lung cancer image datasets.
- Transfer learning using the Xception pre-trained model.
- Training with data augmentation for improved generalization.
- Custom callbacks for learning rate adjustment, early stopping, and checkpointing.
- Visualization of training progress.
- Functions for predicting and displaying classifications for single images.

---

## Dataset Structure
The dataset is expected to have the following structure:
```
dataset/
├── train/
│   ├── normal/
│   ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/
│   ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/
│   ├── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/
├── valid/
│   ├── normal/
│   ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/
│   ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/
│   ├── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/
├── test/
    ├── normal/
    ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/
    ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/
    ├── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/
```

---

## Dependencies
- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pandas

---

## Usage

### Training the Model
1. Prepare your dataset and structure it as shown above.
2. Set the paths to the training, validation, and test folders in the script.
3. Run the script to train the model:
    ```bash
    python train.py
    ```
4. The best model is saved as `best_model.keras`.

### Visualizing Training Progress
The script includes functionality to visualize training and validation loss and accuracy:
- Loss curve
- Accuracy curve

### Predicting Classes
Use the script to predict the class of an individual image:
1. Specify the path to your image in the `img_path` variable.
2. Run the script.
3. The predicted class will be displayed along with the image.

---

## Key Components

### Image Preprocessing
- Images are resized to `350x350` and normalized to the range `[0, 1]`.
- Data augmentation is applied to the training dataset for better generalization.

### Transfer Learning
- The pre-trained Xception model is used as the base.
- A global average pooling layer and a dense softmax layer are added for classification.

### Callbacks
- **ReduceLROnPlateau**: Reduces the learning rate if the loss does not improve.
- **EarlyStopping**: Stops training if the loss does not improve for six consecutive epochs.
- **ModelCheckpoint**: Saves the best model during training.

### Prediction
- Single images are preprocessed and passed through the trained model.
- The class with the highest probability is returned.

---

## Example Results

| Image | Predicted Class |
|-------|-----------------|
| ![Image 1](content/sq.png) | Squamous Cell Carcinoma |
| ![Image 2](content/ad3.png) | Adenocarcinoma |
| ![Image 3](content/l3.png) | Large Cell Carcinoma |
| ![Image 4](content/n8.jpg) | Normal |

---

## Visualization of Training Progress
Example training curves:
- Loss vs Epochs
- Accuracy vs Epochs

---



## Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions or bugs.

---






