# Lung Cancer Classification Using Deep Learning

This repository contains code for classifying lung cancer images into four categories using a convolutional neural network (CNN) with transfer learning. The project employs the Xception model, fine-tuned for this task, leveraging TensorFlow and Keras.This project uses deep learning to classify lung cancer images into the following categories:  
1. **Normal**  
2. **Adenocarcinoma**  
3. **Large Cell Carcinoma**  
4. **Squamous Cell Carcinoma**

Here is a demonstration of the model in action:

<video width="600" controls>
  <source src="results.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Table of Contents
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)


## Dataset

The dataset must be organized as follows:
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

Install the required libraries using:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow keras
```

---
## Project Structure
```
├── Lung_Cancer_Prediction.py  # python code with model training and evaluation
├── README.md                     # Project documentation
├── dataset/                      # Dataset with train/valid/test splits
└── best_model.keras               # Saved best model weights
```
