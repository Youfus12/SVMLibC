SVMLibC is a minimalist implementation of a Support Vector Machine (SVM) written in C. Designed for educational purposes, this library demonstrates the fundamental concepts of SVMs, including data preprocessing, training using Stochastic Subgradient Descent (SGD), and visualization of decision boundaries using Gnuplot.

Table of Contents:

-Project Overview
-Features
-Data Format
-Installation
-Usage
-Visualization
-Classification Summary
-Contributing
-Contact

Project Overview
Support Vector Machines (SVMs) are powerful supervised learning models used for classification and regression tasks. SVMLibC offers a foundational implementation of a linear SVM, enabling users to train models on custom datasets and visualize the results. This project emphasizes understanding the underlying mathematics and algorithmic steps involved in SVMs.

Features
Data Loading and Parsing: Reads datasets from CSV files with customizable feature selection.
Data Shuffling and Splitting: Randomizes data and splits it into training and testing sets.
Feature Scaling: Standardizes features to have zero mean and unit variance.
SVM Training: Implements Stochastic Subgradient Descent (SGD) for training the SVM model.
Prediction and Evaluation: Predicts labels for test data and calculates accuracy.
Visualization: Creates plots of data points, decision boundaries, and margins using Gnuplot.
Classification Summary: Provides a summary of classification results, including misclassifications.
Data Format
Dataset Structure: The program expects a CSV file with four features and a label in the following format:

Structure of the dataset
Feature1,Feature2,Feature3,Feature4,Label
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0

Label Mapping:

0 ➔ +1
1 ➔ -1
Feature Selection: Users can select any two distinct features from the dataset for training and visualization.

Installation
Clone the Repository:

Clone the project repository to your local machine using Git.

Prepare the Dataset:

Ensure your dataset (e.g., irisExt.csv) is placed in the datasets/ directory and follows the specified format.

Usage
Run the Program:

Execute the compiled svm executable.

Select Features:

Choose two distinct features from the available options when prompted.

Training and Evaluation:

The program will load the dataset, preprocess the data, train the SVM model, and evaluate its performance.

Visualization:

After training, a plot will be generated showing the data points, decision boundary, and margins.

Repeat or Exit:

Optionally, train the model with another pair of features or exit the program.

Visualization
SVMLibC utilizes Gnuplot to visualize the results. The generated plot includes:

Data Points:

Positive class (+1) in orange.
Negative class (-1) in black.
Decision Boundary:

Displayed as a green line.
Margins:

Upper and lower margins shown as dashed black lines.
Classification Summary
After each training session, the program provides a summary detailing:

Positive Class (+1):

Points above the margin.
Points within the margin.
Misclassified points.
Negative Class (-1):

Points below the margin.
Points within the margin.
Misclassified points.
Contributing
Contributions are welcome! Whether it's reporting bugs, suggesting features, or improving documentation, your input is valuable. Please feel free to submit issues or pull requests.


Contact:
Younes Chiad
+213777560304
Email: youneschiad@gmail.com
