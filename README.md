# Masked Face Recognition Study
This project aims to study masked face recognition using various methods and algorithms. The study includes the generation of face embeddings using dlib, the evaluation of different classifiers for masked face detection, and the utilization of the LFW dataset for testing.

## Dependencies
To run this project, you need the following dependencies:

Python 3.x
NumPy
matplotlib
scipy
scikit-learn
PIL
OpenCV
PyTorch
torchvision

## Description
The project consists of multiple scripts that serve different purposes:

classifier.py: This script performs a study on masked face detection. It generates face embeddings using dlib, trains different classifiers such as Support Vector Machines (SVM), K-Nearest Neighbors, Gaussian Process, Decision Tree, MLP, Random Forest, AdaBoost, and Quadratic Discriminant Analysis. The classifiers are evaluated using the generated face embeddings and the provided datasets. The script outputs the confusion matrix and precision scores for each classifier.

inference.py: This script implements a mask classifier using a pre-trained ResNet101 model. It takes an image as input, processes it, and predicts whether the person in the image is wearing a mask or not.

label_detect.py: This script classifies multiple images using the mask classifier. It takes multiple images as input, processes each image, predicts the mask-wearing status, and displays the predicted labels on the console.

## Use Case
This project serves as a study to explore masked face recognition methods and algorithms. By evaluating different classifiers and utilizing face embeddings, it aims to identify the best approach for detecting masked faces. The project also includes a mask classifier using a pre-trained ResNet101 model, which can be applied to various applications such as monitoring mask compliance and analyzing mask-wearing behavior.

Please note that the files in the repository represent codes experimented at different times. Therefore, they can only be used as references and cannot be run directly to obtain results. It is advised to review and modify the code based on specific requirements and datasets.

For further inquiries, please contact the developer at megha99.sudhakaran@gmail.com.
