# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           ConfusionMatrixDisplay, precision_score, recall_score, f1_score)
from sklearn.pipeline import Pipeline

data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

# Create dataframe
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("=" * 40)
print(f"■ Dataset Dimensions: {X.shape}")
print(f"■ Available Characteristics: {feature_names}")
print("=" * 40)

# Train-Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"■ Training Variable: {X_train.shape[0]} samples")
print(f"■ Testing Variable: {X_test.shape[0]} samples")
print("=" * 40)

# Pipeline para SVM
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', random_state=42, probability=True))
])

# Optimal hyperparameter search
svm_param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': [0.001, 0.01, 0.1, 1, 'scale']
}

# Best SVM model
svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=5)
svm_grid.fit(X_train, y_train)
best_svm = svm_grid.best_estimator_
y_pred_svm = best_svm.predict(X_test)
y_pred_proba_svm = best_svm.predict_proba(X_test)

# Create a 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D
figura = plt.figure(figsize=(22, 16))
colors = {'setosa':'red', 'versicolor':'green', 'virginica':'blue'}
ax2 = figura.add_subplot(111, projection='3d')
for specie, df in df.groupby('species'):
    ax2.scatter(df['sepal width (cm)'], df['petal width (cm)'], df['petal length (cm)'], c=colors[specie], label=specie, s=40)
ax2.set_xlabel("sepal width")
ax2.set_ylabel("petal width")
ax2.set_zlabel("petal length")
ax2.set_title("Iris Scatter 3D", fontweight='bold')
ax2.legend()

# Create subplots and set the figure size
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=axes[0,0], cmap='Blues')
axes[0,0].set_title('CONFUSION MATRIX', fontweight='bold')

# Correlation matrix located in axes[0,1]
corr_matrix = df[['sepal length (cm)', 'sepal width (cm)', 
                  'petal length (cm)', 'petal width (cm)']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            ax=axes[0,1], fmt='.2f', square=True)
axes[0,1].set_title('CORRELATION MATRIX', fontweight='bold')

# Petal length distribution located in axes[1,0]
sns.boxplot(x='species', y='petal length (cm)', data=df, ax=axes[1,0], 
            hue='species',
            palette=["#E61313", "#0CCF60", "#170DDF"],
            legend=False)
axes[1,0].set_title('PETAL LENGTH DISTRIBUTION BY SPECIES', fontweight='bold')
axes[1,0].set_ylabel('Petal Length (cm)')
axes[1,0].set_xlabel('Specie')

# Sepal width distribution located in axes[1,1]
sns.histplot(data=df, x='sepal width (cm)', hue='species', 
             ax=axes[1,1], palette=['#E61313', '#0CCF60', '#170DDF'], 
             alpha=0.6, element='step', kde=True)
axes[1,1].set_title('SEPAL WIDTH DISTRIBUTION BY SPECIES', fontweight='bold')
axes[1,1].set_xlabel('Sepal Width (cm)')

# Last step of creating subplots
plt.tight_layout()
plt.show()

# Accuracy results
print("=" * 40)
print(f"■ Cross-validation accuracy: {svm_grid.best_score_:.5f}")
print(f"■ Test set accuracy: {accuracy_score(y_test, y_pred_svm):.5f}")
print("=" * 40)

# Metrics by class
precision = precision_score(y_test, y_pred_svm, average=None)
recall = recall_score(y_test, y_pred_svm, average=None)
f1 = f1_score(y_test, y_pred_svm, average=None)

print("CLASSIFICATION REPORT:")
report = classification_report(y_test, y_pred_svm, target_names=target_names, digits=5)
print(f"{report}")
print("=" * 40)

# Create dataframe with predictions
results_df = pd.DataFrame({
    'Real': y_test,
    'Predicción': y_pred_svm,
    'Correcto': y_test == y_pred_svm
})

results_df['Real_Especie'] = results_df['Real'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
results_df['Pred_Especie'] = results_df['Predicción'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"■ NUMBER OF CORRECT PREDICTIONS: {results_df['Correcto'].sum()}/{len(results_df)}")
print(f"■ FINAL ACCURACY: {accuracy_score(y_test, y_pred_svm):.5f}")
print(f"■ BEST HYPERPARAMETERS: {svm_grid.best_params_}")
print("=" * 40)
