# Wine Classification using Naive Bayes

This project classifies wines into three categories using the Naive Bayes classifier. Both Gaussian and Multinomial Naive Bayes classifiers are used to evaluate their performance. The trained models are then used to perform predictions on the test data.

## Requirements

- scikit-learn
- matplotlib
- seaborn
- pandas

## Dataset

The dataset used is the `wine` dataset from `sklearn.datasets`, which contains chemical analysis results of wines grown in a specific region of Italy.

## Steps

1. **Load Dataset**: Load the wine dataset using `load_wine()` from `sklearn.datasets`.
2. **Data Split**: Split the data into training and testing sets using `train_test_split()` from `sklearn.model_selection`.
3. **Gaussian Naive Bayes**: Train and evaluate the model using `GaussianNB()` from `sklearn.naive_bayes`.
4. **Multinomial Naive Bayes**: Train and evaluate the model using `MultinomialNB()` from `sklearn.naive_bayes`.
5. **Confusion Matrix**: Plot the confusion matrices for both classifiers using `confusion_matrix` from `sklearn.metrics` and visualize them using `seaborn`.

## Code

```python
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load dataset
data_set = load_wine()

# Split data into training and testing sets
input = data_set.data
label = data_set.target
x_train, x_test, y_train, y_test = train_test_split(input, label, random_state=10, test_size=0.2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
gaussian_score = gaussian.score(x_test, y_test)
print(f'Gaussian Naive Bayes Score: {gaussian_score}')

# Confusion Matrix for Gaussian Naive Bayes
cn_gaussian = confusion_matrix(y_test, gaussian.predict(x_test))
sns.heatmap(cn_gaussian, cmap='Greens', annot=True, xticklabels=data_set.target_names, yticklabels=data_set.target_names)
plt.title("Gaussian Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

# Multinomial Naive Bayes
multinomial = MultinomialNB()
multinomial.fit(x_train, y_train)
multinomial_score = multinomial.score(x_test, y_test)
print(f'Multinomial Naive Bayes Score: {multinomial_score}')

# Confusion Matrix for Multinomial Naive Bayes
cn_multinomial = confusion_matrix(y_test, multinomial.predict(x_test))
sns.heatmap(cn_multinomial, cmap='Blues', annot=True, xticklabels=data_set.target_names, yticklabels=data_set.target_names
