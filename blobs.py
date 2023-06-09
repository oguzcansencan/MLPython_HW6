import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_curve, auc, mean_absolute_error, mean_squared_error

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

# Create an ensemble of RandomForestClassifier models
n_estimators = 10  # Number of trees in the ensemble
ensemble = []

for _ in range(n_estimators):
    # Create an individual model
    # use different values for three parameters in RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, max_depth=2, max_features=5, random_state=42)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Add the trained model to the ensemble
    ensemble.append(model)

# Make predictions with each model in the ensemble
ensemble_predictions = []
for model in ensemble:
    y_pred = model.predict(X_test)
    ensemble_predictions.append(y_pred)

# Voting for ensemble predictions
ensemble_predictions = np.array(ensemble_predictions)
final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=ensemble_predictions)

# Calculate the accuracy of the ensemble predictions
accuracy = accuracy_score(y_test, final_predictions)
print("Ensemble Accuracy:", accuracy)

######################################################

# Create individual models for the ensemble
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model3 = RandomForestClassifier(n_estimators=100, random_state=42)

# Define the ensemble with different voting methods and weights
ensemble = VotingClassifier(
    estimators=[('model1', model1), ('model2', model2), ('model3', model3)],
    voting='soft',  # Use 'hard' or 'soft' for different voting methods
    weights=[1, 2, 1]  # Assign weights to individual models
)

# Fit the ensemble on the training data
ensemble.fit(X_train, y_train)

# Make predictions with the ensemble
y_pred = ensemble.predict(X_test)

# Calculate the accuracy of the ensemble predictions
accuracy = accuracy_score(y_test, y_pred)
print("Ensemble Accuracy:", accuracy)

######################################################

# Define base estimators with basic parameters
base_estimators = [
    ('rf1', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('rf2', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('rf3', RandomForestClassifier(n_estimators=300, random_state=42))
]

# Compare ensembles with different hyperparameters
for voting in ['hard', 'soft']:
    for weights in [[1, 1, 1], [1, 2, 1], [1, 1, 2]]:
        ensemble = VotingClassifier(
            estimators=base_estimators,
            voting=voting,
            weights=weights
        )

        # Fit the ensemble on the training data
        ensemble.fit(X_train, y_train)

        # Make predictions with the ensemble
        y_pred = ensemble.predict(X_test)

        # Calculate the accuracy of the ensemble predictions
        accuracy = accuracy_score(y_test, y_pred)
        print("Voting:", voting, "Weights:", weights)
        print("Ensemble Accuracy:", accuracy)
        print()


######################################################

# Define a range of n_estimators values
n_estimators_range = [10, 50, 100, 150, 200]

# Initialize lists to store accuracy values for ensembles and individual models
ensemble_accuracies = []
model_accuracies = []

# Iterate over different n_estimators values
for n_estimators in n_estimators_range:
    # Create an ensemble of RandomForestClassifier models
    ensemble = VotingClassifier(
        estimators=[
            ('model1', RandomForestClassifier(n_estimators=n_estimators, random_state=42)),
            ('model2', RandomForestClassifier(n_estimators=n_estimators, random_state=42)),
            ('model3', RandomForestClassifier(n_estimators=n_estimators, random_state=42))
        ],
        voting='soft'
    )

    # Fit the ensemble on the training data
    ensemble.fit(X_train, y_train)

    # Make predictions with the ensemble
    ensemble_predictions = ensemble.predict(X_test)

    # Calculate the accuracy of the ensemble predictions
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    ensemble_accuracies.append(ensemble_accuracy)

    # Create an individual RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions with the model
    model_predictions = model.predict(X_test)

    # Calculate the accuracy of the model predictions
    model_accuracy = accuracy_score(y_test, model_predictions)
    model_accuracies.append(model_accuracy)

    ######################################################
    # Generate the confusion matrix
    cm = confusion_matrix(y_test, ensemble_predictions)
    print("Confusion Matrix:")
    print(cm)

    # Print classification report with precision, recall, and F1 score
    print("Classification Report:")
    print(classification_report(y_test, ensemble_predictions))

    # Compute ROC curve and AUC
    probabilities = ensemble.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probabilities[:, 1])
    roc_auc = auc(fpr, tpr)

######################################################


# Plot the dependencies of n_estimators for ensembles and individual models
plt.plot(n_estimators_range, ensemble_accuracies, label='Ensemble')
plt.plot(n_estimators_range, model_accuracies, label='Individual Model')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('make_blobs - Accuracy vs n_estimators')
plt.legend()
plt.show()
plt.savefig('plots/make_blobs - Accuracy vs n_estimators.png')

######################################################

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the best neural network model from previous stages
best_nn_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)

# Define other classifiers for the ensemble
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create the ensemble using VotingClassifier
ensemble = VotingClassifier(
    estimators=[
        ('nn', best_nn_model),
        ('rf', rf_model)
    ],
    voting='hard'  # Use 'hard' or 'soft' for different voting methods
)

# Fit the ensemble on the training data
ensemble.fit(X_train, y_train)

# Make predictions with the ensemble
y_pred = ensemble.predict(X_test)

# Calculate the accuracy of the ensemble predictions
accuracy = accuracy_score(y_test, y_pred)
print("Ensemble Accuracy:", accuracy)


######################################################

# Define the basic model (e.g., MLPClassifier)
model_basic = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', random_state=42)

# Define the ensemble with the basic model
ensemble = VotingClassifier(estimators=[('basic', model_basic)], voting='hard')

# Fit the basic model on the data
model_basic.fit(X, y)

# Fit the ensemble on the data
ensemble.fit(X, y)

# Create a meshgrid to plot the decision boundaries
h = 0.02  # Step size in the mesh
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain predictions for the meshgrid points using the basic model
Z_basic = model_basic.predict(np.c_[xx.ravel(), yy.ravel()])
Z_basic = Z_basic.reshape(xx.shape)

# Obtain predictions for the meshgrid points using the ensemble
Z_ensemble = ensemble.predict(np.c_[xx.ravel(), yy.ravel()])
Z_ensemble = Z_ensemble.reshape(xx.shape)

# Plot the decision boundaries
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_basic, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired', edgecolors='k')
plt.title('Decision Boundaries - Basic Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_ensemble, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired', edgecolors='k')
plt.title('Decision Boundaries - Ensemble')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
plt.savefig('plots/make_blobs - Decision Boundaries.png')

######################################################

# Define the basic model (e.g., MLPClassifier)
model_basic = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)

# Define the ensemble with the basic model
ensemble = VotingClassifier(estimators=[('basic', model_basic)], voting='hard')

# Fit the basic model on the data
model_basic.fit(X, y)

# Fit the ensemble on the data
ensemble.fit(X, y)

# Calculate predictions for the basic model and ensemble
y_pred_basic = model_basic.predict(X)
y_pred_ensemble = ensemble.predict(X)

# Calculate the shift using mean absolute error (MAE)
shift_basic = mean_absolute_error(y, y_pred_basic)
shift_ensemble = mean_absolute_error(y, y_pred_ensemble)

# Calculate the variation using mean squared error (MSE)
variation_basic = mean_squared_error(y, y_pred_basic)
variation_ensemble = mean_squared_error(y, y_pred_ensemble)

print("Shift - Basic Model:", shift_basic)
print("Shift - Ensemble:", shift_ensemble)
print("Variation - Basic Model:", variation_basic)
print("Variation - Ensemble:", variation_ensemble)

######################################################
