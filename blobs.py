import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_curve, auc

matplotlib.use('TKAgg', force=True)

n_samples_1 = 1000
n_samples_2 = 100
centers = [[0.0, 0.0], [2.0, 2.0]]
clusters_std = [1.5, 0.5]
X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers, cluster_std=clusters_std, random_state=0, shuffle=False)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', random_state=42, early_stopping=True)
model.fit(X_train_scaled, y_train)

print("Hidden Layer Size: (100, 50)")

accuracy = model.score(X_val_scaled, y_val)
print("Validation Accuracy:", accuracy)

test_accuracy = model.score(X_test_scaled, y_test)
print("Test Accuracy:", test_accuracy)
#######################################################

model = MLPClassifier(hidden_layer_sizes=(5, 5), activation='relu', random_state=42, early_stopping=True)
model.fit(X_train_scaled, y_train)

print("Hidden Layer Size: (5, 5)")

accuracy = model.score(X_val_scaled, y_val)
print("Validation Accuracy:", accuracy)

test_accuracy = model.score(X_test_scaled, y_test)
print("Test Accuracy:", test_accuracy)

########################################################

# Retrieve the loss values
loss_curve = model.loss_curve_

# Print the loss values at the first and last iterations
print("Loss at first iteration:", loss_curve[0])
print("Loss at last iteration:", loss_curve[-1])

# Plot the loss curve
plt.plot(loss_curve)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("make_blobs - Loss Curve")
plt.show()

######################################################

# Calculate the average accuracy for the training set
train_accuracy = model.score(X_train_scaled, y_train)
print("Training Accuracy:", train_accuracy)

# Calculate the average accuracy for the test set
test_accuracy = model.score(X_test_scaled, y_test)
print("Test Accuracy:", test_accuracy)

######################################################

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate precision, recall, and F1 score
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Compute ROC curve and AUC
y_prob = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('make_blobs - Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
plt.savefig("plots/make_blobs - roc_curve.png")

######################################################
