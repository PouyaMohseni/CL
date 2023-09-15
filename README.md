# Grapevine Leaves Classification Research

## Introduction

This research project is focused on the classification of grapevine leaves, with the primary goal of identifying the most effective classification model among various methods. The dataset used in this study consists of images of grapevine leaves categorized into five distinct classes, each containing one hundred images of a specific grape leaf type. The ultimate objective is to select the best-performing model based on its accuracy in classifying these images.

### Dataset
You can access the dataset used in this research via this [link](https://www.muratkoklu.com/datasets/Grapevine_Leaves_Image_Dataset.zip).

## Classification Methods

In this section, we provide an overview of the eight different classification methods explored in this research. Each method incorporates distinct approaches and techniques, such as feature extraction, transfer learning, feature selection, augmentation, and the use of various neural network architectures.

### Model 1: Simple CNN
- A basic Convolutional Neural Network (CNN) with five layers.
- Architecture:
  - Conv2D(9, 3, padding='same', activation='relu')
  - MaxPooling2D()
  - Conv2D(32, 3, padding='same', activation='relu')
  - MaxPooling2D()
  - Conv2D(64, 3, padding='same', activation='relu')
  - MaxPooling2D()
  - Flatten()
  - Dense(128, activation='relu')
- Trained for 45 epochs.

### Model 2: Augmented Data
- Data augmentation techniques applied, including horizontal flips, translations, zooming, and rotations.
- A sequential neural network used for image augmentation, with primary layers superimposed.
- Trained for 150 epochs, benefiting from the augmented data.

### Model 3: Fine-Tuned MobileNetv2
- Utilizes a pre-trained MobileNetv2 model with fine-tuning.
- Adds a classification layer while maintaining input size constraints.
- Input images are not augmented.

### Model 4: Augmentation + MobileNetv2
- Augmentation layers introduced before inputting images into the pre-trained MobileNetv2 network.
- Enables the use of more training epochs without overfitting concerns.

### Model 5: VGG16 Feature Extractor
- Employs VGG16 as a feature extractor.
- Utilizes the output of VGG16's last layer as input for Random Forest Classifier (RFC) and Support Vector Classifier (SVC) with different parameters.

### Model 6: Dimension Reduction with PCA
- Applies Principal Component Analysis (PCA) to reduce feature dimensions.
- Multiple PCA runs retain varying numbers of attributes: 365, 200, 100, and 50.

### Model 7: Enhanced PCA
- Builds upon Model 6 with an augmented input data strategy.
- Adds a neural network to augment input data, resulting in 1500 input features.
- Explores thirteen different feature dimension reduction modes.

### Model 8: Autoencoder Feature Extraction
- Utilizes autoencoders to identify optimal VGG16 output features.
- Comprises four phases: data augmentation, VGG16 attribute generation, encoder-based dimension reduction, and classification using Random Forest or SVM.

## Conclusion

At the conclusion of this research, the best-performing model is determined to be an SVM model, which leverages input parameters derived from VGG16 (Model 5). To validate the model's performance, cross-fold validation is implemented, and the resulting accuracies are visualized using a confusion matrix. It's worth noting that static augmentation is employed due to the necessity of specifying most of the data in K-fold Cross validation. This research provides valuable insights into the classification of grapevine leaves, showcasing the effectiveness of various approaches and techniques.
