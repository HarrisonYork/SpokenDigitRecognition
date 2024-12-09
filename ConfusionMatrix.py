import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# data for confusion matrix
confusion_matrix = [
    [146, 1, 1, 12, 3, 10, 10, 13, 0, 24],
    [1, 191, 6, 3, 3, 3, 0, 2, 4, 7],
    [0, 1, 203, 3, 1, 5, 0, 1, 6, 0],
    [40, 10, 34, 123, 0, 0, 0, 11, 2, 0],
    [0, 10, 0, 0, 153, 0, 0, 48, 0, 9],
    [14, 0, 10, 1, 6, 100, 0, 41, 0, 48],
    [41, 0, 16, 3, 0, 4, 138, 11, 0, 7],
    [6, 3, 1, 2, 57, 11, 0, 122, 0, 18],
    [0, 11, 37, 2, 0, 0, 0, 1, 164, 5],
    [0, 0, 8, 6, 7, 19, 4, 43, 0, 133]
]

# K-means or EM
method = 'KM'
# number of iterations of method
iters = 105
# type of covariance
cov = 'Tied Spherical'
# accuracy from notebook output
overall_accuracy = 66.95

plt.figure(figsize=(9, 9))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Oranges', cbar=False)

plt.xlabel('Predicted Digit', size=16)
plt.ylabel('True Digit', size=16)
plt.title(f'Confusion Matrix: {method} {iters} iters., {cov} Covariance, {overall_accuracy}% Accuracy', size=16)
plt.show()
