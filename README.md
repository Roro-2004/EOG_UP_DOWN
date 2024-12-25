EOG UP & DOWN Classifier
This project is a graphical user interface (GUI) application that uses machine learning to classify Electrooculography (EOG) data as either "UP" or "DOWN" movements. The application allows users to train a model with labeled data, then predict movements on new, unseen data.

Table of Contents
Overview
Requirements
Installation
Usage
Functions
License
Overview
The application processes EOG data using the following steps:

Preprocessing: Filters the signal, removes outliers, normalizes data, and resamples it.
Feature Extraction: Extracts wavelet-based features from the EOG signal for use in machine learning.
Model Training: The user can upload labeled "Up" and "Down" movement data to train a K-Nearest Neighbors (KNN) model.
Prediction: After training, the model can predict whether a new test signal represents an "Up" or "Down" movement.
The application is built with Tkinter for the graphical interface, and it utilizes Scikit-learn, SciPy, PyWavelets, and other libraries for signal processing and machine learning.

Requirements
Python 3.x
Libraries:
numpy
scipy
pywt (PyWavelets)
sklearn
matplotlib
tkinter (for GUI)
scipy.stats (for statistical features like z-scores, skew, and kurtosis)
Installation
To run this application, follow these steps:

Install Python 3.x: Make sure you have Python 3.x installed on your system.

Install dependencies: You can install the required libraries by running the following command:

bash
Copy code
pip install numpy scipy pywt scikit-learn matplotlib
Download the code: Clone or download this repository to your local machine.

bash
Copy code
git clone <repository_url>
Run the application: After installation, you can run the application directly by running the following command in your terminal or command prompt:

bash
Copy code
python eog_classifier.py
Usage
Training the Model:

Click the "Browse" button to select the file containing "Up" movement data.
Click the "Browse" button to select the file containing "Down" movement data.
Click "Train Model" to train the model using the selected data.
A message box will confirm if the model was trained successfully.
Predicting Movement:

Click the "Browse" button to select the test data file.
Click "Predict" to classify the test data.
The prediction (either "UP" or "DOWN") will be displayed in a message box along with the confidence of the prediction.
Functions
butter_bandpass(lowcut, highcut, fs, order=4)
This function generates the coefficients for a Butterworth bandpass filter with a given low and high cutoff frequency and sampling frequency.

preprocessing(file_path, fs=176, lowcut=0.5, highcut=20)
This function processes the raw EOG data:

Removes outliers using z-scores.
Centers the data by removing the mean.
Applies a bandpass filter.
Normalizes the signal.
Resamples the data and removes null values.
extract_wavelet_features(signal)
This function extracts statistical features from the EOG signal using wavelet decomposition:

Applies multiple wavelet families (e.g., 'db1', 'db2') and decomposes the signal into coefficients.
Calculates statistical measures like mean, standard deviation, skewness, kurtosis, median, and interquartile range (IQR).
EOGClassifier
This class encapsulates the KNN classifier logic:

train(up_file, down_file): Trains the classifier using labeled "Up" and "Down" movement data.
predict(test_file): Makes predictions on a test file, returning the predicted movement ("UP" or "DOWN") along with confidence.
EOGInterface
This class creates the Tkinter-based GUI, which includes:

Input fields for selecting training and test files.
Buttons to train the model and make predictions.
