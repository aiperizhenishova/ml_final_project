# ==============================================
# WATER QUALITY PREDICTION PROJECT
# ==============================================

# 1. PREPARE PROBLEM
# ==============================================

# a) Load required libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score


# b) Load dataset
print("\n" + "="*50)
print("1. PREPARE PROBLEM")
print("="*50)

file_path = 'water_potability.csv'
data = pd.read_csv(file_path)
print("\nDataset loaded from '{}':".format(file_path))
print("Rows: {}, Columns: {}".format(data.shape[0], data.shape[1]))
print("Columns:", list(data.columns))

# Handle missing values (basic mean imputation for 3 specific columns)
for col in ['ph', 'Sulfate', 'Trihalomethanes']:
    data[col] = data[col].fillna(data[col].mean())

# ==============================================
# 2. EXPLORATORY DATA ANALYSIS (Summarize Data)
# ==============================================

print("\n" + "="*50)
print("2. Summarize Data")
print("="*50)

# only 39% of the water is potable.
print("\n=== CLASS DISTRIBUTION ===")
print(data['Potability'].value_counts())
print("\nPotable Water Ratio: {:.1%}".format(data['Potability'].mean()))

print("\n=== DATASET OVERVIEW ===")
print("Shape:", data.shape)
print("\nData types:\n", data.dtypes)

print("\n=== DESCRIPTIVE STATISTICS ===")
print(data.describe().transpose())

# how much the distributions are skewed
print("\n=== FEATURE SKEWNESS ===")
print(data.skew().sort_values(ascending=False))

# Histograms
data.hist(figsize=(12, 8))
plt.suptitle('Feature Distributions', y=1.02)
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# Boxplots
data.plot(kind='box', subplots=True, layout=(4, 3), figsize=(12, 8))
plt.suptitle('Box Plots of Features', y=1.02)
plt.tight_layout()
plt.savefig('boxplots.png')
plt.close()

# ==============================================
# 3. DATA PREPROCESSING (Prepare Data)
# ==============================================

print("\n" + "="*50)
print("3. Prepare Data")
print("="*50)

print("\n=== HANDLING MISSING VALUES ===")
print("Missing values before cleaning:")
print(data.isnull().sum())

# Advanced imputation
imputer = IterativeImputer(random_state=42)
features = data.columns.drop('Potability')
X = data[features]
y = data['Potability']
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print("\nMissing values after cleaning:", X_imputed.isnull().sum().sum())

# Feature engineering
# to improve the training of the model
print("\n=== CREATING NEW FEATURES ===")
X_imputed['TDS_ratio'] = X_imputed['Solids'] / \
    (X_imputed['Conductivity'] + 1e-6)
X_imputed['Disinfection_score'] = np.log1p(
    X_imputed['Chloramines'] * X_imputed['Trihalomethanes'])
X_imputed['Mineral_index'] = X_imputed['Hardness'] * \
    X_imputed['Sulfate'] / 1000
print("Added 3 new engineered features:")
print("1. TDS_ratio = Solids / (Conductivity + 1e-6)")
print("2. Disinfection_score = log1p(Chloramines * Trihalomethanes)")
print("3. Mineral_index = (Hardness * Sulfate) / 1000")
print(list(X_imputed.columns))

# Scaling
#  to make it more balanced and easier for models to learn from.
print("\n=== SCALING AND NORMALIZATION ===")
scaler = PowerTransformer(method='yeo-johnson')
X_scaled = scaler.fit_transform(X_imputed)
print("Applied Yeo-Johnson power transformation")

# SMOTE
# SMOTE — Synthetic Minority Over-sampling Technique (artificially created examples)
print("\n=== CLASS BALANCING ===")
print("Before SMOTE - Class counts:", np.bincount(y))
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
print("After SMOTE - Class counts:", np.bincount(y_res))

# ==============================================
# 4. EVALUATE ALGORITHMS
# ==============================================

print("\n" + "="*50)
print("4. ALGORITHM EVALUATION")
print("="*50)

# a) Split-out validation dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.15, random_state=42)
print(
    f"\nData split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")

# b) Test options and evaluation metric
print("\nEvaluation metric: Accuracy & F1-score")

# c) Spot Check Algorithms
print("\n=== SPOT CHECK ALGORITHMS ===")

models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42))
]

results = []
for name, model in models:
    # Train model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    f1 = classification_report(y_test, y_pred, output_dict=True)[
        'macro avg']['f1-score']
    results.append((name, acc, f1))
    print(f"{name:20} | Accuracy: {acc:.2%} | F1-score: {f1:.2%}")

# d) Compare Algorithms
print("\n=== ALGORITHM COMPARISON ===")
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1-score'])
results_df = results_df.sort_values('Accuracy', ascending=False)
print(results_df.to_string(index=False))

# Visual comparison
plt.figure(figsize=(10, 6))
results_df.set_index('Model').plot(kind='bar', rot=45)
plt.title('Algorithm Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('algorithm_comparison.png')
plt.close()

# ==============================================
# 5. IMPROVE ACCURACY (UPDATED)
# ==============================================

print("\n" + "="*50)
print("5. Improve Accuracy")
print("="*50)

# Select best performing algorithm (Gradient Boosting)
print("\nSelected best performing algorithm: RandomForestClassifier")

# a) Algorithm Tuning with GridSearchCV
print("\n=== HYPERPARAMETER TUNING ===")
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters found:")
print(grid_search.best_params_)

# b) Ensembles
print("\n=== FINAL ENSEMBLE MODEL ===")
best_model = grid_search.best_estimator_

# Оценка финальной модели на тестовых данных
final_preds = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_preds)
final_f1 = f1_score(y_test, final_preds)

print(f"\nFinal Model Accuracy: {final_accuracy * 100:.2f}%")
print(f"Final Model F1-Score: {final_f1 * 100:.2f}%")


# ==============================================
# 6. MODEL DEPLOYMENT
# ==============================================

print("\n" + "="*50)
print("6. Finalize Model")
print("="*50)

# Saving preprocessing components
# joblib.dump saves your model or data to a file so you don’t have to train it again every time.
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'transformer.pkl')
joblib.dump(list(X_imputed.columns), 'feature_names.pkl')

print("\nPreprocessing components saved successfully!")
print("- Imputer: imputer.pkl")
print("- Scaler: transformer.pkl")
print("- Feature names: feature_names.pkl")

# Train the model on the entire dataset
best_model.fit(X_res, y_res)  # Use the entire dataset after SMOTE

# Save the trained model
joblib.dump(best_model, 'final_model.pkl')

print("\nModel trained on the entire dataset and saved as 'final_model.pkl'")

print("\n" + "="*50)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("="*50)
