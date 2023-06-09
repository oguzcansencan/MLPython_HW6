import numpy as np

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


X, y = make_circles(500, factor=0.1, noise=0.1)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', random_state=42, early_stopping=True)
model.fit(X_train_scaled, y_train)

accuracy = model.score(X_val_scaled, y_val)
print("Validation Accuracy:", accuracy)

test_accuracy = model.score(X_test_scaled, y_test)
print("Test Accuracy:", test_accuracy)
#######################################################

model = MLPClassifier(hidden_layer_sizes=(5, 5), activation='relu', random_state=42, early_stopping=True)
model.fit(X_train_scaled, y_train)

accuracy = model.score(X_val_scaled, y_val)
print("Validation Accuracy:", accuracy)

test_accuracy = model.score(X_test_scaled, y_test)
print("Test Accuracy:", test_accuracy)

########################################################
