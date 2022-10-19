# CL
Models for grapevine leaves classification 

This research has been conducted to find the best classification model to classify images of the data presented grapevine leaves. There are five classes and each constains one hundred images of a specific type of grape leaf. In the following, we discuss eight different methods for classification, and based on the accuracy, the best model is choosed.

Different approachs such as feature extraction, trasfer learning, feature selection, augmentation, encoders etc. were used in different models. Here different models are briefly summerized:

Dataset: https://www.muratkoklu.com/datasets/Grapevine_Leaves_Image_Dataset.zip

Model 1:
In this method, we create a simple CNN with 5 layers as follows. (Model is trained with 45 epochs) 
  layers.Conv2D(9, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)

Model 2:
For this method, the data of the most part is augmented. To augment images, we perform several different fuvtions such as horizontal filp, trasfering, zooming and rotation.
Augmentation of images is preformed by the help of a sequential neural network; we create a new network so that several primary layers are superimposed on the image augmentation and later layers according to the primary network architecture. Therefore, in this architecture, we fit the initial model on the augmented images. (In this method, unlike the previous method, we use 150 epochs, because the model made of augmented images is less threatened by overfit.)

Model 3:
In this method, we use the pre-trained MobileNetv2 model after being fine tuned. So that a classification layer is added and only the weights leading to the last layer are traind. For this method, it is necessary that the number of pixels in length and width be equal to the input of the pre-traind moreover, the images should be three-channeled, which we obtain by resizing the input images. It should be noted that in this method, the input images are not augmented.

Model 4:
In this method, we try to involve a layer with the ability to augment data in the third method. So that before the images enter the pre-trained MobileNetv2 network, they pass through layers to be augmented. By using this method, we can increase the number of epochs without fear of having an overfited model.

Model 5:
First, we use the state-of-art 16VGG as feature extractor. Then we use the output of the last layer of the pre-learned model for the input of the RFC and the SVC with different parameters.

Model 6:
In the previous method, 25088 (the size of VGG16 classes) features are extracted and given to the classifiers. This amount is much lower than 365, which is the number of Images (without augmentation), so PCA method is used to reduce the dimensions of the model output -as a feature selector. It should be noted that due to the input limitation, a maximum of 365 different attributes can be kept. We run PCA four times and each time we keep 365, 200, 100 and 50 attributes respectively.

Model 7:
The weak part of the previous model was not having a large number of inputs, which made the comparison of PCA dimensions meaningless. In the seventh method, we follow the sequence of the sixth method, with the difference that we attach a neural network to the top of VGG16 to augment the input data and input the previous data four times (augment the data four times), so we have the 1500 most input. We check thirteen different modes for the number of features. Some of the results of the implementation of these dimension reductions on the six previously introduced models are shown in the figure below, and the accuracy diagram of the models according to the dimension reduction is shown below:
![Uploading image.png…]()

Model 8:
Now we try to use autoencoder to identify the best output features of VGG16; In this way, we give the output of VGG16 to an autoencoder and train the model so that it can reconstruct the output classes. In fact, we use four separate phases to meet the needs. The first phase is for data augmentation, the second phases is VGG16 for generating attributes, the encoder model is for dimension reduction, and finally the random forest model or SVM for classification.


Conclusion:
At the end, we select the best built model and implement cross-fold validation on it, and then polt the obtained accuracies in the confusion matrix. The best model was an SVM model whose input parameters came from the VGG16 model (5th model). (Because the most of the data must be specified in K-fold Cross, the previous extension cannot be used to augment the data, so we use static augmentaion.)
