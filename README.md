# Detection of Breast Cancer Using Decision Tree Classifier
 This project illustrates the application of K-Nearest Neighbour which is an supervised classification model!

# Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Exploratory Data Analsysis](#exploratory-data-analysis)
4. [Model Application](#model-application)
5. [Evaluation](#evaluation)
6. [Libraries Used](#libraries-used)
7. [How to Run](#how-to-run)
8. [Acknowledgements](#acknowledgements)

## Overview
 This projects helps us to classify the `breast cancer` cases as *malignant* and *benign* using the **K-nearest neighbour** model.

## Dataset
 It utilizes the standard dataset from **scikit learn datasets** which is the `load_breast_cancer` dataset.

## Exploratory Data Analsysis
 In this step, we check the following:
1. The **shape** of the **dataset**.
2. The description/information related to the dataset.
3. We plot a correlation heatmap in order to understand deeply about the dataset.

## Model Application
 Then we directly apply the model using **scikit-learn** *framework* from the standard `sklearn.neighbors` library.

## Evaluation
 Later, we evaluate the model using several metrics such as accuracy score, confusion matrix and the classification report.
Here, we found out that

* The *accuracy* of **KNN** : `0.9912`

* Classification Matrix of KNN Model : 
               precision    recall  f1-score   support

   malignant       1.00      0.98      0.99        52
      benign       0.98      1.00      0.99        62

    accuracy                           0.99       114
   macro avg       0.99      0.99      0.99       114
weighted avg       0.99      0.99      0.99       114
Then at last, we use the seaborn and matplotlib.pyplot library in order to evaluate the confusion matrix of the applied model!

## Libraries Used
              We used several libraries which include
              numpy, pandas, matplotlib, seaborn, scikit-learn

## How to Run 
1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone [https://github.com/Suchendra13/Breast_Cancer_KNN.git](https://github.com/Suchendra13/Breast_Cancer_KNN.git)
    cd Breast_Cancer_KNN
    ```
2.  **Ensure you have Jupyter Notebook installed** or use a compatible IDE (e.g., VS Code with Jupyter extensions).
3.  **Install the required Python libraries**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
4.  **Open the Jupyter Notebook**:
    ```bash
    jupyter notebook Breast_Cancer_KNN.ipynb
    ```
5.  **Run all cells** in the notebook.

## Acknowledgements
 We have used the standard *scikit-learn dataset* i.e. `load_breast_cancer` for the educational purpose.
