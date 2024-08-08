# Machine Learning-Based Phishing Website Detection

In this project, I implemented various machine learning models to classify websites as legitimate or phishing. The analysis involves data preprocessing, model training, evaluation, and comparison of multiple classifiers including Decision Tree, Naive Bayes, Bagging, Boosting, Random Forest, Artificial Neural Networks, and Support Vector Machines.

## Dataset

The dataset contains 2000 rows and 26 columns with the following characteristics:
- All columns are of integer or numerical data types.
- The dataset includes NA values ranging from 11 to 27 across columns, except for the predictor `A01` and the target variable `Class`.
- The target variable `Class` has two values: "0" (legitimate) and "1" (phishing), with an uneven distribution.

## Preprocessing

1. **Handling NA values**: Removed rows with NA values, retaining 1556 rows and 26 columns.
2. **Removing Low Variance Predictors**: Dropped predictors with more than 99% values in a single category.
3. **Encoding Categorical Variables**: Converted predictors with less than 10 unique values to factor columns.
4. **Data Splitting**: Split data into training (70%) and testing (30%) sets.

## Models

The following models were implemented:

1. **Decision Tree**
2. **Naïve Bayes**
3. **Bagging**
4. **Boosting**
5. **Random Forest**
6. **Artificial Neural Network (ANN)**
7. **Support Vector Machine (SVM)**

## Results

- **Random Forest**: Highest accuracy and AUC, demonstrating robust performance in distinguishing between legitimate and phishing websites.
- **Decision Tree**: Lowest accuracy, primarily due to the assumption of independence between predictors.
