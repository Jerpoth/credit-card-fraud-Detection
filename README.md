# credit-card-fraud-Detection
 Credit Card Fraud Detection is a critical application of data science and machine learning. It involves identifying and preventing fraudulent transactions in credit card transactions. Here's a detailed description of the problem and how it is typically approached:
Importing Dependencies: The code starts by importing various Python libraries that are commonly used in machine learning, including NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn (for machine learning algorithms).

Loading the Dataset: The dataset is loaded from a CSV file named 'creditcard.csv'. This dataset likely contains information about credit card transactions.

Data Exploration:

credit_card.head(): Displays the first 5 rows of the dataset.
credit_card.tail(): Displays the last 5 rows of the dataset.
credit_card.info(): Provides information about the dataset, including the number of rows, columns, and data types.
credit_card.isnull().sum(): Checks for missing values in each column. There are no missing values in this dataset.
Data Distribution:

credit_card['Class'].value_counts(): Shows the distribution of classes (0 for non-fraudulent transactions, 1 for fraudulent transactions). It reveals a significant class imbalance, with non-fraudulent transactions being the majority.
Data Visualization:

credit_card.hist(): Plots histograms for each parameter in the dataset, providing a visual overview of the data distribution.
Data Separation:

The dataset is split into two subsets, one containing non-fraudulent transactions (denoted as 'legit') and the other containing fraudulent transactions (denoted as 'fraud').
legit.shape and fraud.shape display the number of rows and columns in each subset.
Statistical Analysis:

Descriptive statistics (mean, standard deviation, min, max, quartiles) are computed for the 'Amount' column for both legitimate and fraudulent transactions.
Creating a Balanced Dataset:

A balanced dataset is created by randomly selecting a sample of legitimate transactions (492 samples) to match the number of fraudulent transactions. This balances the class distribution.
Data Splitting:

The balanced dataset is split into training and testing sets using the train_test_split function.
Data Standardization:

Data standardization is performed to ensure that all features have the same scale. The StandardScaler from scikit-learn is used to standardize the data.
Model Training and Evaluation:

Two machine learning models are trained and evaluated:
Logistic Regression: The code uses logistic regression to build a classification model. The model is evaluated using metrics such as accuracy, precision, recall, F1-score, mean absolute error, and mean squared error.
XGBoost Classifier: Another classification model is built using the XGBoost algorithm, and its performance is evaluated in a similar manner.
Algorithm Comparison:

A comparison of the two models is presented in a tabular format. The table includes accuracy, precision, recall, and F1-score for each model. XGBoost appears to outperform logistic regression in terms of accuracy and precision.
Conclusion:

The code concludes by suggesting that XGBoost is a slightly better choice for credit card fraud detection due to its higher precision and accuracy.
This project can be considered as a basic example of binary classification using machine learning for fraud detection. The goal is to detect fraudulent credit card transactions accurately while minimizing false positives.
