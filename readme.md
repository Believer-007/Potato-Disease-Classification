

This project is designed to detect diseases in potato leaves using machine learning techniques. The dataset used is the PlantVillage Dataset from Kaggle, which includes images of potato leaves affected by various diseases like Early Blight, Late Blight, and Healthy leaves.

The project leverages deep learning, specifically Convolutional Neural Networks (CNN), to classify the health of potato leaves. The goal is to provide an automated system for farmers and agronomists to detect and diagnose potato diseases early, which can help in preventing crop loss and improving yield.

Dataset:
-The dataset used for training and testing the model is sourced from the Plant Village dataset on Kaggle. This dataset contains a large number of images of potato leaves affected by different diseases, as well as healthy leaves.

Model Architecture:
-The core of the system is a deep convolutional neural network (CNN) architecture.

Training Process

Data Preprocessing:
-Image resizing and normalization
-Data augmentation techniques (e.g., rotation, flipping, zooming)
-Splitting the dataset into training and validation sets.

Model Training:
-Using a suitable loss function (e.g., categorical cross-entropy)
-Optimizing the model using an appropriate optimizer (e.g., Adam)
-Training the model for a specified number of epochs

Model Evaluation:
-Evaluating the model's performance on the validation set using metrics like accuracy, precision, recall, and F1-score
-Visualizing the model's predictions on sample images