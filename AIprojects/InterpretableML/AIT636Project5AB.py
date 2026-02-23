import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

df = pd.read_csv('C:/Users/Administrator/Downloads/pima-indians-diabetes.csv', index_col=0)
feature_names = df.columns[:-1]
# print(df.head())

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(df.drop('target', axis=1))
scaled_features = scaler.transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
# print(df_feat.head())

sns.pairplot(df, hue='target')
plt.show()

# Split the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3, stratify=df['target'], random_state=42)

# Apply Least Squares
from sklearn import linear_model
# Ordinary least squares
# clf = linear_model.LinearRegression()
# Ridge regression
'''clf = linear_model.Ridge(alpha=0.5, # Regularization strength: Larger values specify stronger regularization.
                 random_state=0, # # The seed of the pseudo random number generator to use when shuffling the data.
                 )'''
# Lasso regression has a tendency to prefer solutions with fewer parameter values, effectively reducing the number of variables.
clf = linear_model.Lasso(alpha=0.1, # Regularization strength: Larger values specify stronger regularization. Default is 1.
                     random_state=0, # The seed of the pseudo random number generator that selects a random feature to update.
                     )

clf = clf.fit(x_train, y_train)

# Predictions
predictions_test = clf.predict(x_test)
class_names = [0, 1]
predictions_test[predictions_test <= 0.5] = 0
predictions_test[predictions_test > 0.5] = 1
predictions_test = predictions_test.astype(int)

# Display confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=class_names)
confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_names)
confusion_matrix_display.plot()
plt.show()

# Report Overall Accuracy, precision, recall, F1-score
print(metrics.classification_report(
    y_true=y_test,
    y_pred=predictions_test,
    target_names=list(map(str, class_names)),
    zero_division=0 # Whenever number is divided by zero, instead of nan, return 0
))

# Optimize alpha
# Optimize alpha for BOTH Lasso and Ridge

alpha_values = np.arange(0, 2, 0.2)

# Store results separately
overall_accuracies_lasso = []
overall_accuracies_ridge = []

class_names = [0, 1]

# -----------------------------
# LASSO OPTIMIZATION
# -----------------------------
print("\n================ LASSO =================")

for i in alpha_values:
    print('alpha is:', i)

    clf = linear_model.Lasso(alpha=i, random_state=0)
    clf = clf.fit(x_train, y_train)

    predictions_test = clf.predict(x_test)

    predictions_test[predictions_test <= 0.5] = 0
    predictions_test[predictions_test > 0.5] = 1
    predictions_test = predictions_test.astype(int)

    overall_accuracies_lasso.append(
        metrics.accuracy_score(y_true=y_test, y_pred=predictions_test)
    )

    confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=class_names)
    confusion_matrix_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=class_names
    )
    confusion_matrix_display.plot()
    plt.title(f"Lasso (alpha={i})")
    plt.show()

    print(metrics.classification_report(
        y_true=y_test,
        y_pred=predictions_test,
        target_names=list(map(str, class_names)),
        zero_division=0
    ))

# -----------------------------
# RIDGE OPTIMIZATION
# -----------------------------
print("\n================ RIDGE =================")

for i in alpha_values:
    print('alpha is:', i)

    clf = linear_model.Ridge(alpha=i, random_state=0)
    clf = clf.fit(x_train, y_train)

    predictions_test = clf.predict(x_test)

    predictions_test[predictions_test <= 0.5] = 0
    predictions_test[predictions_test > 0.5] = 1
    predictions_test = predictions_test.astype(int)

    overall_accuracies_ridge.append(
        metrics.accuracy_score(y_true=y_test, y_pred=predictions_test)
    )

    confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=class_names)
    confusion_matrix_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=class_names
    )
    confusion_matrix_display.plot()
    plt.title(f"Ridge (alpha={i})")
    plt.show()

    print(metrics.classification_report(
        y_true=y_test,
        y_pred=predictions_test,
        target_names=list(map(str, class_names)),
        zero_division=0
    ))

# -----------------------------
# Plot BOTH Accuracy Curves
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(alpha_values, overall_accuracies_lasso,
         color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red',
         markersize=10)

plt.title('Lasso: Accuracy vs. alpha_value')
plt.xlabel('alpha-value')
plt.ylabel('Accuracy')

plt.show()


plt.figure(figsize=(10, 6))

plt.plot(alpha_values, overall_accuracies_ridge,
         color='green', linestyle='dashed',
         marker='o', markerfacecolor='black',
         markersize=10)

plt.title('Ridge: Accuracy vs. alpha_value')
plt.xlabel('alpha-value')
plt.ylabel('Accuracy')

plt.show()