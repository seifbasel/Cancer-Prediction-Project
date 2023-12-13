import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Load data from a CSV file
file_path = 'Battery_RUL.csv'  # Replace with the actual path to your dataset
df = pd.read_csv(file_path)

# Common features and target variable
features_to_exclude = ['RUL']
X = df.drop(features_to_exclude, axis=1)
y = df['RUL']

# Downscale the dataset using random sampling
downscale_fraction = 0.7
X_downscaled, _, y_downscaled, _ = train_test_split(X, y, test_size=downscale_fraction, random_state=42)

# Split the downscaled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_downscaled, y_downscaled, test_size=0.2, random_state=42)

# Define preprocessing steps in a pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns)
    ])

# Define the pipeline with Random Forest Regressor
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))
])

# Fit the pipeline
pipeline_rf.fit(X_train, y_train)

# Make predictions using Random Forest Regressor
y_pred_rf = pipeline_rf.predict(X_test)

# k-Nearest Neighbors (KNN) Regression
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train)
y_pred_knn_reg = knn_reg.predict(X_test)

# Support Vector Machine (SVM) Regression
svm_reg = SVR(kernel='linear')
svm_reg.fit(X_train, y_train)
y_pred_svm_reg = svm_reg.predict(X_test)

# Decision Trees Regression
pre_pruned_tree_reg = DecisionTreeRegressor(max_depth=3)
pre_pruned_tree_reg.fit(X_train, y_train)

unpruned_tree_reg = DecisionTreeRegressor()
unpruned_tree_reg.fit(X_train, y_train)

# Calculate metrics for each model
metrics = {
    'Random Forest Regression': {
        'MSE': mean_squared_error(y_test, y_pred_rf),
        'R-squared': r2_score(y_test, y_pred_rf)
    },
    'k-Nearest Neighbors (KNN) Regression': {
        'MSE': mean_squared_error(y_test, y_pred_knn_reg),
        'R-squared': r2_score(y_test, y_pred_knn_reg)
    },
    'Support Vector Machine (SVM) Regression': {
        'MSE': mean_squared_error(y_test, y_pred_svm_reg),
        'R-squared': r2_score(y_test, y_pred_svm_reg)
    },
    'Decision Tree (Pre-Pruned) Regression': {
        'MSE': mean_squared_error(y_test, pre_pruned_tree_reg.predict(X_test)),
        'R-squared': r2_score(y_test, pre_pruned_tree_reg.predict(X_test))
    },
    'Decision Tree (Unpruned) Regression': {
        'MSE': mean_squared_error(y_test, unpruned_tree_reg.predict(X_test)),
        'R-squared': r2_score(y_test, unpruned_tree_reg.predict(X_test))
    }
}

# Print metrics for each model
for model, metrics_dict in metrics.items():
    print(f'{model} Metrics:')
    print(f'Mean Squared Error (MSE): {metrics_dict["MSE"]:.2f}')
    print(f'R-squared: {metrics_dict["R-squared"]:.2f}')
    print('\n')

# Visualization: Predicted vs Actual Values
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_rf, label='Random Forest Regression', alpha=0.7)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Regression: Predicted vs Actual Values')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred_knn_reg, label='k-Nearest Neighbors (KNN) Regression', alpha=0.7)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('KNN Regression: Predicted vs Actual Values')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred_svm_reg, label='Support Vector Machine (SVM) Regression', alpha=0.7)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('SVM Regression: Predicted vs Actual Values')
plt.legend()

plt.show()

# Visualization: Decision Tree (Pre-Pruned)
plt.figure(figsize=(12, 6))
plot_tree(pre_pruned_tree_reg, filled=True, feature_names=X.columns)
plt.title("Pre-Pruned Decision Tree Regression")
plt.show()

# Visualization: Decision Tree (Unpruned)
plt.figure(figsize=(12, 6))
plot_tree(unpruned_tree_reg, filled=True, feature_names=X.columns)
plt.title("Unpruned Decision Tree Regression")
plt.show()

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_curve, auc
# import matplotlib.pyplot as plt

# # Load data from a CSV file
# file_path = 'cancer_prediction_dataset.csv'
# df = pd.read_csv(file_path)

# # Common features and target variable
# features_to_exclude = ['Cancer']
# X = df.drop(features_to_exclude, axis=1)
# y = df['Cancer']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize/Normalize features for Linear Regression
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # k-Nearest Neighbors
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)
# y_pred_knn = knn.predict(X_test)
# accuracy_knn = accuracy_score(y_test, y_pred_knn)
# print(f'kNN Accuracy: {accuracy_knn:.2f}')

# # Support Vector Machine (SVM)
# svm_classifier = SVC(kernel='linear', C=1)
# svm_classifier.fit(X_train, y_train)
# y_pred_svm = svm_classifier.predict(X_test)
# accuracy_svm = accuracy_score(y_test, y_pred_svm)
# print(f'SVM Accuracy: {accuracy_svm:.2f}')

# # Random Forest Classifier
# random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
# random_forest.fit(X_train, y_train)
# y_pred_rf = random_forest.predict(X_test)
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print(f'Random Forest Accuracy: {accuracy_rf:.2f}')

# # Linear Regression
# linear_reg_model = LinearRegression()
# linear_reg_model.fit(X_train_scaled, y_train)
# y_pred_linear_reg = linear_reg_model.predict(X_test_scaled)
# mse = mean_squared_error(y_test, y_pred_linear_reg)
# r2 = r2_score(y_test, y_pred_linear_reg)
# print(f'Mean Squared Error: {mse:.2f}')
# print(f'R-squared: {r2:.2f}')

# # Decision Trees
# pre_pruned_tree = DecisionTreeClassifier(max_depth=3)
# pre_pruned_tree.fit(X_train, y_train)
# unpruned_tree = DecisionTreeClassifier()
# unpruned_tree.fit(X_train, y_train)

# # Plot the pre-pruned tree
# plt.figure(figsize=(12, 6))
# plot_tree(pre_pruned_tree, filled=True, feature_names=X.columns, class_names=[str(c) for c in df['Cancer'].unique()])
# plt.title("Pre-Pruned Decision Tree")
# plt.show()

# # Plot the unpruned tree
# plt.figure(figsize=(12, 6))
# plot_tree(unpruned_tree, filled=True, feature_names=X.columns, class_names=[str(c) for c in df['Cancer'].unique()])
# plt.title("Unpruned Decision Tree")
# plt.show()

# # Function to plot ROC curve
# def plot_roc(model, X_test, y_test, label):
#     fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

# # ROC curves for all classifiers
# plt.figure(figsize=(10, 8))

# # k-Nearest Neighbors
# plot_roc(knn, X_test, y_test, 'k-Nearest Neighbors')

# # Support Vector Machine (SVM)
# plot_roc(svm_classifier, X_test, y_test, 'SVM')

# # Random Forest Classifier
# plot_roc(random_forest, X_test, y_test, 'Random Forest')

# # Display the plot
# plt.title('Receiver Operating Characteristic (ROC) Curves')
# plt.legend(loc='lower right')
# plt.show()
