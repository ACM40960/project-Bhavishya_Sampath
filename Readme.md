# AI Image Recognition Challenge: Newfoundland vs Bear

## Project Summary
This project is focused on developing and evaluating various Convolutional Neural Network (CNN) models to accurately differentiate between images of Newfoundland dogs and bears. It explores multiple approaches, including transfer learning with VGG16, custom-designed Sequential CNNs, and methods for handling binary classification and imbalanced datasets. The models are assessed using metrics such as accuracy, recall, AUC, ROC curves, and confusion matrices, with additional validation through K-fold cross-validation to ensure robustness.

## Setup Requirements
Before beginning this project, make sure you have the following setup:

Python Installation: Ensure you have Python installed on your system. You can download and install it from the official Python website.

  pip install tensorflow numpy matplotlib keras
1. GPU Acceleration (Optional): For faster training, use platforms like Google Colab or Kaggle that provide free GPU access. Enable GPU support in Colab under runtime settings or - 
  configure it in Kaggle notebooks.
  
2. Image Dataset:  Newfoundland and Bear image datasets
3. Code Editor: Use an editor or IDE such as Jupyter Notebook, VS Code, or PyCharm.
   
 

## Dataset Preparation
The dataset consists of balanced images equally split between Newfoundland dogs and bears. The dataset is divided into:

Training Set (60%): Used to train the model to learn distinguishing features.
Validation Set (20%): Used during training to evaluate and adjust hyperparameters.
Testing Set (20%): Used to assess final model performance post-training.
An imbalanced dataset was also explored to test the model's robustness under unequal class distributions.

Training Set:
Bear: 218 images
Newfoundland: 201 images
Validation Set:
Bear: 200 images
Newfoundland: 194 images
Testing Set:
Bear: 200 images
Newfoundland: 197 images


An imbalanced dataset was also explored to test the model's robustness under unequal class distributions.

## Directory Structure

 NewfoundlandvsBear/
│

├── Bear/

│   ├── train/

│   ├── validation/

│   └── test/

│
└── Newfoundland/

    ├── train/
    ├── validation/
    └── test/
## Methodology
## 1. VGG16 Transfer Learning with K-Fold Cross-Validation
The VGG16 model, pre-trained on ImageNet, was employed as the base model for binary classification between Newfoundland dogs and bears. Custom layers were added on top of the pre-trained base to tailor it to the specific task. To ensure robustness and generalizability, K-fold cross-validation was used.

### Key Components:

Frozen Base Layers: The pre-trained layers of VGG16 were frozen to retain the features learned from ImageNet.
Custom Fully Connected Layers: These layers were added to adapt the model for binary classification, concluding with a softmax activation function for outputting class probabilities.
K-Fold Cross-Validation: The dataset was split into k folds, with the model trained and validated across different subsets to evaluate its performance comprehensively.
Evaluation Metrics: Accuracy, precision, recall, F1-score, AUC, ROC curves, and confusion matrices were used to evaluate the model's performance across folds.
## 2. Sequential CNN
A custom Sequential CNN was built for binary classification. This model comprises several convolutional layers for feature extraction, followed by fully connected layers for decision-making.

### Key Components:

Convolutional Layers: These layers extract essential features from the input images.
MaxPooling Layers: These layers reduce the spatial dimensions of the feature maps, retaining critical information.
Flatten Layer: Converts 2D feature maps into a 1D vector, making it suitable for the fully connected layers.
Fully Connected Layers: These layers learn high-level representations from the features extracted.
Dropout Layer: This layer prevents overfitting by randomly deactivating neurons during training.
Output Layer: Utilizes a softmax activation function to output class probabilities, which is crucial for binary classification.
## 3. Binary Classification using VGG16
The VGG16 model was fine-tuned specifically for binary classification tasks, where it needed to distinguish between two classes: Newfoundland and Bear.

### Key Components:

Frozen Base Layers: The pre-trained layers of VGG16 were retained to utilize the rich feature representations learned from ImageNet.
Custom Layers for Binary Output: A dense layer with a sigmoid activation function was added to output a probability score between 0 and 1, used for binary classification.
Binary Classification: The sigmoid function in the final output layer allowed the model to effectively classify images into one of the two categories.
## 4. Binary Classification using Sequential CNN
A custom Sequential CNN model was built and optimized for binary classification, specifically to differentiate between Newfoundland dogs and bears.

### Key Components:

Sequential Model Architecture: The architecture included multiple convolutional layers, pooling layers, and a dense output layer with a sigmoid activation function.
Binary Classification: The model was trained to output a single probability score per image, indicating the likelihood of the image belonging to a particular class.
Training Strategy: The model was trained with binary cross-entropy loss, optimized using the Adam optimizer.
## 5. ResNet50 Transfer Learning
ResNet50, another pre-trained model, was similarly employed using transfer learning. The ResNet50 model was fine-tuned for the binary classification task.

### Key Components:

Frozen Base Layers: The pre-trained ResNet50 layers were retained to leverage learned features.
Custom Layers: Added to modify the model for binary classification, ending with a sigmoid output layer for final predictions.
Binary Classification: The sigmoid function is used in the output layer to distinguish between the two classes.
## 6. Imbalanced Dataset Handling
Class imbalance was addressed using the following strategies:

Class Weights: Adjusted the importance of each class during training to ensure balanced performance across both classes.
Augmented Datasets: Data augmentation was used to artificially balance the dataset, helping the model perform better under imbalanced conditions.
Evaluation Metrics: Special emphasis was placed on recall and AUC to ensure that the model accurately identified the minority class in the imbalanced dataset.
Data Augmentation
Data augmentation techniques were applied to enhance the training dataset's diversity and improve the model’s robustness:

Rescale: Normalizes pixel values to a [0, 1] range.
Random Rotation: Simulates different image orientations to increase dataset variability.
Width and Height Shifts: Introduces randomness in image positioning to account for potential shifts.
Horizontal Flip: Creates mirror images, further increasing dataset diversity.
Model Evaluation
The models were evaluated using several key metrics:

Accuracy: Measures the overall correctness of the model's predictions.
Precision: Indicates the accuracy of the positive predictions made by the model.
Recall: Measures the model's ability to correctly identify all positive instances.
AUC (Area Under the Curve): Evaluates the model's performance across all classification thresholds, providing a comprehensive view of its effectiveness.
ROC Curve: The ROC curve visualizes the model's ability to distinguish between classes by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
Confusion Matrix: The confusion matrix was used to visualize the model’s classification performance by displaying the counts of true positives, true negatives, false positives, and false negatives.
## Confusion Matrices
Confusion matrices were generated for each model to evaluate their performance in distinguishing between Newfoundland and Bear classes. The confusion matrix provides a detailed breakdown of the model's predictions:

True Positives (TP): Correctly predicted Newfoundland images as Newfoundland.
True Negatives (TN): Correctly predicted Bear images as Bear.
False Positives (FP): Incorrectly predicted Bear images as Newfoundland.
False Negatives (FN): Incorrectly predicted Newfoundland images as Bear.
The confusion matrix helps identify where the model may be struggling, such as predicting one class more accurately than the other or having a high number of false positives or false negatives. This information is crucial for improving the model's accuracy and robustness.

## Results
VGG16 Transfer Learning with K-Fold Cross-Validation: Achieved the highest accuracy of 96%, demonstrating strong generalization and robustness across different folds.
Sequential CNN: Showed a moderate performance with an accuracy of 86%, effectively learning to classify Newfoundland dogs and bears.
Binary Classification using VGG16: Also performed well, with an accuracy of 95%, confirming the effectiveness of using VGG16 for binary classification tasks.
Binary Classification using Sequential CNN: Achieved an accuracy of 62%, indicating challenges in generalizing across the binary classification task.
ResNet50 Transfer Learning: Showed a slightly improved performance over the Sequential CNN with an accuracy of 68%.
Imbalanced Dataset Handling: Techniques such as class weighting and data augmentation significantly improved the model’s ability to handle imbalanced datasets, enhancing overall robustness.
## Web Application

A Flask-based web application was developed to provide a user-friendly interface for making predictions using the trained models. Users can upload an image, and the app will predict whether the image is of a Newfoundland or a bear. The app enhances the user experience by changing the background and providing tailored messages based on the prediction result.
## Conclusion
This project successfully demonstrates the capability of CNNs and transfer learning in solving complex image classification tasks. By employing data augmentation, class weights, and carefully chosen architectures, the models achieved high accuracy in distinguishing between Newfoundland dogs and bears, even in the presence of class imbalance. The use of ROC curves and confusion matrices provided deeper insights into the models' performance, highlighting their strengths and areas for improvement.

## Future Work
Potential future improvements include:

Exploring Deeper Architectures: Testing deeper models like ResNet101 or more advanced architectures such as EfficientNet.
Hyperparameter Tuning: Further optimization of learning rates, batch sizes, and other hyperparameters to improve model performance.
Expanding the Dataset: Including additional images or more classes to increase the model’s generalization capability.
Domain Adaptation: Applying the models to related classification tasks, such as distinguishing between other similar-looking species or objects.
## Acknowledgments
This project was completed with the help of various online resources and datasets. Special thanks to TensorFlow and Keras for providing the tools needed to build and train the models.
## Credits
I have colloborated with Sampath Vadrevu - 23200373 on this project.
## References
1. J.S, N., Tharun, K. & Savya, S. G., 2024. Deep Learning Approaches To Image-Based Species Identification. International Conference on Integrated Circuits and Communication Systems, pp. 1-7.
2. Li, J. et al., 2023. Deep learning for visual recognition and detection of aquatic animals: A review. Reviews in Aquaculture, 15(2), pp. 409-433.
3. Upadhyay, A., Prajapati, V. & Shinde, A., 2023. Ant Species Recognition using Convolutional Neural Network. Bhopal, India, IEEE International Students' Conference on Electrical, Electronics and Computer Science, pp. 1-5.
4. [https://dl.acm.org/doi/fullHtml/10.1145/3582197.3582216](https://dl.acm.org/doi/fullHtml/10.1145/3582197.3582216)
TensorFlow Documentation
  5.Keras Documentation


   6.VGG16 Paper

   7.ResNet Paper

   8.Kaggle

   9.Github
