# Bias-Variance Analysis on the Pima Indians Diabetes Dataset

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

A hands-on study of the **bias–variance tradeoff** on the Pima Indians Diabetes
dataset. Five classifiers (Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting, SVM) are trained, diagnosed for underfitting/overfitting,
and then tuned with `GridSearchCV` to find a healthier point on the
bias–variance curve.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Load & Split the Data](#1-load--split-the-data)
  - [2. Train a Baseline Logistic Regression](#2-train-a-baseline-logistic-regression)
  - [3. Run the Bias–Variance Diagnostic](#3-run-the-biasvariance-diagnostic)
  - [4. Tune Models with GridSearchCV](#4-tune-models-with-gridsearchcv)
  - [5. Plot Learning Curves](#5-plot-learning-curves)
- [Results](#results)
- [Key Takeaways](#key-takeaways)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The notebook (`diabetes.ipynb`) is organized into five sections that mirror a
realistic ML workflow:

1. **Data Exploration & Visualization** — shape, summary stats, class balance,
   correlation heatmap, per-feature distributions, outlier detection, pairplot.
2. **Model Training & Prediction** — scaled features, baseline Logistic
   Regression, confusion matrix, ROC/AUC.
3. **Bias–Variance Analysis** — quantifies bias (`1 - train_score`) and variance
   (`std` of CV scores) across five models and flags each as underfitting,
   overfitting, or balanced.
4. **Model Adjustment & Optimization** — `GridSearchCV` over regularization
   strength, tree depth, leaf size, ensemble size, etc.
5. **Final Predictions & Summary** — picks the best model by test accuracy and
   prints a written summary of the tradeoff.

## Dataset

| Property | Value |
| --- | --- |
| Source | Pima Indians Diabetes Database |
| Rows | 768 |
| Features | 8 numerical (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) |
| Target | `Outcome` (0 = no diabetes, 1 = diabetes) |
| Class balance | ~65% negative / ~35% positive |

Place `diabetes.csv` next to the notebook before running the cells.

## Project Structure

```
Bias-Variance-Analysis/
├── diabetes.ipynb        # Main analysis notebook
├── diabetes.csv          # Pima Indians Diabetes dataset
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT license
└── README.md             # This file
```

## Installation

```bash
git clone https://github.com/<your-username>/Bias-Variance-Analysis.git
cd Bias-Variance-Analysis

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Minimum dependencies:

```text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
```

Launch the notebook:

```bash
jupyter notebook diabetes.ipynb
```

## Usage

The snippets below mirror the notebook so you can paste them into a fresh
script or REPL.

### 1. Load & Split the Data

```python
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 2. Train a Baseline Logistic Regression

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### 3. Run the Bias–Variance Diagnostic

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def diagnose(model, X_tr, X_te, y_tr, y_te, name):
    model.fit(X_tr, y_tr)
    train = model.score(X_tr, y_tr)
    test  = model.score(X_te, y_te)
    cv    = cross_val_score(model, X_tr, y_tr, cv=5)

    bias     = 1 - train
    variance = cv.std()

    print(f"{name:>20s} | train={train:.3f} test={test:.3f} "
          f"bias={bias:.3f} variance={variance:.3f}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM":                 SVC(random_state=42),
}

for name, m in models.items():
    diagnose(m, X_train_scaled, X_test_scaled, y_train, y_test, name)
```

### 4. Tune Models with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

rf_params = {
    "n_estimators":      [50, 100, 200],
    "max_depth":         [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params, cv=5, scoring="accuracy", n_jobs=-1,
)
rf_grid.fit(X_train_scaled, y_train)

print("Best params:", rf_grid.best_params_)
print(f"Best CV:    {rf_grid.best_score_:.4f}")
print(f"Test:       {rf_grid.score(X_test_scaled, y_test):.4f}")
```

### 5. Plot Learning Curves

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    rf_grid.best_estimator_, X_train_scaled, y_train,
    cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
)

plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train")
plt.plot(train_sizes, test_scores.mean(axis=1),  "o-", label="CV")
plt.xlabel("Training examples"); plt.ylabel("Accuracy")
plt.title("Learning Curve — Random Forest")
plt.legend(); plt.grid(True); plt.show()
```

## Results

Numbers from the notebook (untuned baselines, `test_size=0.2`, `random_state=42`):

| Model | Train Acc | Test Acc | CV Mean | Bias | Variance | Diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 0.7704 | 0.7532 | 0.7606 | 0.230 | 0.030 | High bias |
| Decision Tree | 1.0000 | 0.7468 | 0.7198 | 0.000 | 0.052 | High variance |
| Random Forest | 1.0000 | 0.7208 | 0.7753 | 0.000 | 0.034 | Overfits the training set |
| Gradient Boosting | 0.9381 | 0.7403 | 0.7704 | 0.062 | 0.024 | Balanced |
| SVM | 0.8339 | 0.7338 | 0.7687 | 0.166 | 0.024 | Slight bias |

After tuning, the **Optimized Decision Tree** reached **0.7792 test accuracy**
with `max_depth=5`, `min_samples_leaf=10` — the cleanest example of trading a
bit of variance for substantially better generalization.

## Key Takeaways

- **High train, low test** is the classic signature of variance — fix it with
  regularization, pruning, or more data.
- **Low train and low test** is bias — fix it by adding features or reducing
  regularization.
- Ensemble methods (Random Forest, Gradient Boosting) sit closer to the sweet
  spot out of the box, but still benefit from tuning depth and leaf size.
- Learning curves are the single best visual diagnostic — converged curves at
  a high score mean you are done.

## Contributing

PRs and issues are welcome. If you spot a bug, want to add another classifier,
or improve the plots, open an issue first so we can align on scope.

## License

This project is released under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2026 Arnav Pareek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
