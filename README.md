##FashionVision — Comparing Dense and Convolutional Neural Networks on Fashion-MNIST

FashionVision is a deep learning project that explores how different neural network architectures—Dense Neural Networks (DNNs) and Convolutional Neural Networks (CNNs)—perform on the Fashion-MNIST dataset, a modern benchmark for image classification of clothing items.

The goal of this project is to compare model accuracy, training efficiency, and visual feature extraction capabilities, demonstrating the advantages of convolutional architectures in handling image data.

## Dataset

Fashion-MNIST is a dataset of 70,000 grayscale images (28×28 px) across 10 fashion categories:

T-shirt/top

Trouser

Pullover

Dress

Coat

Sandal

Shirt

Sneaker

Bag

Ankle boot

## Features

Implemented two deep learning models:

Dense Neural Network (Fully Connected Layers)

Convolutional Neural Network (CNN)

Compared training/validation accuracy and loss

Generated confusion matrix and classification report

Visualized sample predictions with true vs predicted labels

Highlighted how CNN learns spatial patterns for better feature extraction

## Model Architecture
1. Dense Neural Network (DNN)

Flatten layer → Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)

Simpler and faster, but limited spatial understanding

2. Convolutional Neural Network (CNN)

Conv2D(32,3×3) → MaxPooling → Conv2D(64,3×3) → Flatten → Dense(128) → Dense(10, Softmax)

Captures edges, shapes, and textures more efficiently

## Results Summary
Model	Test Accuracy	Test Loss
Dense Neural Network	~86%	~0.45
Convolutional Neural Network	~91%	~0.29

(Results may slightly vary per run due to random initialization.)

## Visualizations

Accuracy vs. Loss curves for both models

Confusion Matrix highlighting misclassifications

16-sample prediction grid (True vs Predicted labels)

## Key Insights

CNN significantly outperformed the Dense model due to its ability to learn spatial hierarchies in images.

Dense networks can handle tabular or flattened inputs well but lose contextual relationships between pixels.

Proper regularization (Dropout, Batch Normalization) and small kernel sizes improved generalization.

## 🔧 Tech Stack

Python, TensorFlow/Keras, NumPy, Matplotlib, Seaborn

Jupyter Notebook / Google Colab

## Learning Outcome

This project deepened understanding of:

How convolution layers enhance feature detection

The trade-off between model simplicity and performance

Designing reproducible ML experiments with clear metrics and visuals

## Next Steps

Implement Data Augmentation to further boost generalization

Experiment with Transfer Learning (e.g., MobileNet, ResNet50)

Extend to real-world fashion datasets with higher resolution images

Author
Manveer Singh
mssandhuu05@gmail.com

