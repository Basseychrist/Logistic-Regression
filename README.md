# Logistic Regression Lab — Notebook Explanation and Exercise Solutions

This repository contains the notebook `Logistic-Regression-v1.ipynb`, a hands-on lab that demonstrates using Logistic Regression (scikit-learn) to predict customer churn. This README walks through the notebook step-by-step, explains the key concepts, and provides worked solutions for the Practice Exercises included at the end of the notebook.

---

## Contents of this README

- High-level summary
- Required libraries and environment
- Notebook walkthrough (section-by-section explanation)
- How to run the notebook locally
- Answers to Practice Exercises (a–e) with reproducible code snippets and interpretation
- Next steps and references

---

## High-level summary

The notebook loads a Telco churn dataset, selects a subset of features, preprocesses them (standardization), trains a Logistic Regression classifier, and evaluates the model using log-loss (binary cross-entropy). It also visualizes the learned coefficients and asks a set of practice questions about how adding/removing features affects log loss.

Goals:

- Show a minimal, reproducible ML workflow: load -> preprocess -> train -> predict -> evaluate.
- Give intuition about feature importance via logistic regression coefficients.
- Explore how feature additions/removals change predictive performance (log loss).

---

## Required libraries / environment

The notebook installs and uses these packages:

- Python (3.8+ recommended)
- numpy
- pandas
- scikit-learn
- matplotlib

The notebook contains pip install lines. To reproduce on Windows PowerShell, you can run (in a notebook cell or PowerShell):

```powershell
python -m pip install numpy==2.2.0 pandas==2.2.3 scikit-learn==1.6.0 matplotlib==3.9.3
```

Or run the pip lines inside the notebook (they are already present in the first code cell).

---

## Notebook walkthrough (section-by-section)

1. Title, objectives and imports

- The notebook begins with learning objectives and installs required packages.
- Key imports: pandas, numpy, train_test_split, LogisticRegression, StandardScaler, log_loss, matplotlib.

2. Loading the Telco Churn data

- The notebook loads a CSV from a URL. This dataset contains one row per customer and many fields including demographic and service indicators such as `callcard`, `wireless`, and `equip`.

3. Feature selection and target

- For this lab the base input features are: `['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']` and the target is `churn`.
- The notebook also converts `churn` to integer type to meet scikit-learn expectations.

4. Building X and y

- X is the features array (NumPy) and y is the churn label array.

5. Standardization

- StandardScaler is used to standardize X (zero mean, unit variance). Standardizing helps many models converge and ensures features are comparable in scale.

6. Train/Test split

- The notebook uses `train_test_split` with `test_size=0.2` and `random_state=4` to create reproducible splits.

7. Model training and prediction

- Uses `LogisticRegression` (scikit-learn default) and trains on the training split.
- Uses `predict` for class labels and `predict_proba` for class probabilities.

8. Feature coefficients

- `LR.coef_[0]` gives the learned weight for each feature. Positive coefficients increase the log-odds of class 1 (churn), negative decrease them.

9. Performance evaluation

- Primary metric in this lab: log loss (binary cross-entropy). Lower log loss is better.

10. Practice Exercises

- The notebook asks how log loss changes when adding/removing features: `callcard`, `wireless`, `equip`, `income`, and `employ`.

---

## How to run the notebook locally

1. Open `Logistic-Regression-v1.ipynb` in Jupyter Notebook / JupyterLab / VS Code.
2. Run the cells in order. The notebook includes the pip installs in a first cell (you may skip reinstalling if packages are already installed).
3. If you want to reproduce the practice exercise numbers below, run the provided code snippets for each exercise (examples included below).

Tip: Ensure your machine has internet access to download the CSV dataset from the URL used in the notebook.

---

## Answers to Practice Exercises (a–e)

Below are reproducible code snippets and the expected log loss results as provided by the notebook. You can copy each code snippet into a new notebook cell (or into the existing notebook replacing the feature selection line) and run it.

Common boilerplate used in each snippet (unchanged across a–e):

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)
# Ensure churn is integer
churn_df['churn'] = churn_df['churn'].astype('int')

def compute_logloss(feature_list):
    X = np.asarray(churn_df[feature_list])
    y = np.asarray(churn_df['churn'])
    X_norm = StandardScaler().fit(X).transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)
    LR = LogisticRegression().fit(X_train, y_train)
    yhat_prob = LR.predict_proba(X_test)
    return log_loss(y_test, yhat_prob)

base_features = ['tenure','age','address','income','ed','employ','equip']
```

a) Add `callcard` to the original set of input features.

```python
features_a = base_features + ['callcard']
loss_a = compute_logloss(features_a)
print('log loss (with callcard):', loss_a)
```

Expected / notebook answer: 0.6039104035600186

Interpretation: Adding `callcard` changed the model probability outputs and resulted in the shown log loss. If this number is lower than the baseline log loss it indicates the new feature helped; if higher it hurt performance.

b) Add `wireless` to the original set of input features.

```python
features_b = base_features + ['wireless']
loss_b = compute_logloss(features_b)
print('log loss (with wireless):', loss_b)
```

Expected / notebook answer: 0.7227054293985518

Interpretation: In the notebook's run this increased log loss relative to the baseline, which suggests `wireless` either adds noise or has weaker discriminatory power for churn in this dataset.

c) Add both `callcard` and `wireless` to the input features.

```python
features_c = base_features + ['callcard', 'wireless']
loss_c = compute_logloss(features_c)
print('log loss (with callcard and wireless):', loss_c)
```

Expected / notebook answer: 0.7760557225417114

Interpretation: In the provided run adding both made log loss worse than some earlier variants. This can happen when features are correlated or add noise.

d) Remove the feature `equip` from the original feature set.

```python
features_d = ['tenure','age','address','income','ed','employ']  # removed 'equip'
loss_d = compute_logloss(features_d)
print('log loss (without equip):', loss_d)
```

Expected / notebook answer: 0.5302427350245369

Interpretation: Removing `equip` changed log loss to the shown value. If log loss decreased after removing a feature, that often indicates the feature may have been noisy or led to worse probability calibration.

e) Remove `income` and `employ` from the original set.

```python
features_e = ['tenure','age','address','ed','equip']  # removed 'income' and 'employ'
loss_e = compute_logloss(features_e)
print('log loss (without income and employ):', loss_e)
```

Expected / notebook answer: 0.6529317169884828

Interpretation: Removing two features produced the shown result. Judging feature utility should be done based on metrics like log loss as well as domain interpretation.

Notes about reproducibility

- The exact numeric values above come from running the notebook with `random_state=4` and the dataset at the provided URL. If you rerun on a different random_state or dataset snapshot you may get slightly different values.

---

## Verified results (run locally)

I ran the included script `compute_exercises.py` locally (on your workspace) to verify the log-loss values reported in the notebook. Below are the exact numbers produced on my run (Python executable used: `C:/Python313/python.exe` in your environment):

Results:

- base: 0.6257718410257235
- a) with `callcard`: 0.6039104035600186
- b) with `wireless`: 0.7227054293985518
- c) with `callcard` and `wireless`: 0.7760557225417115
- d) without `equip`: 0.5302427350245369
- e) without `income` and `employ`: 0.6529317169884828

These match the expected answers listed in the Practice Exercises section.

If you want these numbers recomputed on a different random seed or with cross-validation, I can add cells to do that as well.

---

## Cross-Validation and Regularization (what I added to the notebook)

I added a short example cell to the notebook that performs a regularization sweep over the inverse regularization strength parameter C (L2 penalty) using 5-fold stratified cross-validation. The cell:

- Loads the same dataset used in the lab and applies `StandardScaler` to the selected features.
- Uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=4)` to produce stable splits.
- For each C in a logarithmic range (`np.logspace(-4, 4, 15)`), it obtains cross-validated predicted probabilities using `cross_val_predict(..., method='predict_proba')` and computes `log_loss` on the full dataset using those probabilities.
- Plots Log Loss vs C (log-scale for the x-axis) and prints the C value that gave the minimum log loss.

Why this is useful:

- Regularization (controlled by C in scikit-learn where smaller C = stronger regularization) penalizes large weights and can reduce overfitting.
- Cross-validation gives a more robust estimate of out-of-sample performance than a single train/test split.

How to run it:

1. Open `Logistic-Regression-v1.ipynb` and run all cells (the new CV/reg cell is at the end).
2. The cell will produce a plot and print the best C and its corresponding cross-validated log-loss.

Notes and next steps:

- If you want, I can add an L1-penalty sweep, compare solvers (liblinear vs lbfgs), or run a small grid-search with `GridSearchCV` that optimizes log-loss directly.
- The cell requires internet access to fetch the dataset from the URL used in the notebook.

## Minimal baseline / sanity check

If you want a quick baseline log loss for just the original features, compute:

```python
base_loss = compute_logloss(base_features)
print('base log loss:', base_loss)
```

Compare this value with the exercise results to see whether adding/removing features helped.

---

## Short interpretations & tips

- Logistic Regression coefficients are linear weights on standardized inputs; large absolute coefficients indicate stronger influence on log-odds.
- Log loss is sensitive to predicted probabilities — well-calibrated probabilities reduce log loss.
- Adding features can increase or decrease log loss depending on whether the feature provides signal or noise, and whether it is correlated with other features.
- Feature selection (regularization, cross-validation, and domain knowledge) can help avoid overfitting and high log loss.

---

## Next steps

- Try L2 or L1 regularization (penalty parameter in LogisticRegression) and measure log loss across a sweep of C values using cross-validation.
- Try other metrics (ROC-AUC, accuracy, precision/recall) depending on how you want to trade off false positives and false negatives.
- Add calibration steps (Platt scaling or isotonic regression) if you need better probability estimates.

---

## References

- scikit-learn LogisticRegression docs: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- Binary cross-entropy / log loss explanation: https://en.wikipedia.org/wiki/Cross_entropy

---

If you'd like, I can also:

- run the notebook cells here to compute and verify the numeric answers (I can run the code and confirm the reported log loss numbers), or
- add a new notebook cell that programmatically computes all five exercise variants and prints the results.

Let me know which option you prefer and I will run it and update this README with the verified numbers.
