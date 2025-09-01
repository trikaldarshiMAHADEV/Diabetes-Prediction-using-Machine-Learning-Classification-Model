Diabetes Prediction Model - 

* This project aims to predict the onset of diabetes in patients based on diagnostic measurements. The analysis uses a dataset with nine columns and 768 entries, including features such as 
Pregnancies, Glucose, Blood Pressure, BMI, Insulin, Age, DiabetesPedigreeFunction, SkinThickness and a target variable called 'Outcome'.
* The pipeline covers EDA, outlier handling, feature engineering, scaling, model training, hyperparameter tuning (GridSearchCV), and evaluation withconfusion matrices and classification
  reports.

1. Data -

* Original shape: (768, 9)
* After outlier filtering: (760, 9)
* After one-hot encoding of engineered features: (760, 19)
* Train/test split: 80/20 (random_state=0, no stratify) → test size 152 rows
* Class distribution (on your processed data): 0 = 65.10%, 1 = 34.90% (non-diabetic vs diabetic)

2. Key Plots & Artifacts -

* Correlation heatmap, box/violin plots, missingness bar plot (missingno), and ROC curves for multiple models.

3. Models & Tuning -

* Trained models include:
* Logistic Regression
*KNN
*SVM (RBF) — tuned with GridSearchCV (best used: C=10, gamma=0.01, probability=True)
*Decision Tree 
* Random Forest — tuned (e.g., criterion='entropy', max_depth=15, max_features=0.75, min_samples_leaf=2, min_samples_split=3, n_estimators=130)
* Gradient Boosting — tuned with GridSearchCV (various loss, learning_rate, n_estimators)
*XGBoost

* SVM (RBF, C=10, γ=0.01)
Train 0.873 → Test 0.914
Confusion: [[91, 7], [6, 48]]
Class 1 — Precision 0.87, Recall 0.89, F1 0.88
* Gradient Boosting (tuned)
Train 0.979 → Test 0.901
Confusion: [[90, 8], [7, 47]]
Class 1 — Precision 0.85, Recall 0.87, F1 0.86
* Random Forest (tuned)
Train 0.993 → Test 0.888
Confusion: [[88, 10], [7, 47]]
Class 1 — Precision 0.82, Recall 0.87, F1 0.85
* XGBoost
Train 0.979 → Test 0.888
Confusion: [[89, 9], [8, 46]]
Class 1 — Precision 0.84, Recall 0.85, F1 0.84
* Decision Tree
Train 1.000 → Test 0.868
Confusion: [[85, 13], [7, 47]]
Class 1 — Precision 0.78, Recall 0.87, F1 0.82

* Out of all these Models SVM come out with hightest accuracy=91.1% almost then GradientBossting=90%

4. On ROC–AUC

-> From  ROC plots and tuned classifiers
* SVM (RBF, C=10, γ=0.01) → AUC ≈ 0.96 (highest)
* Gradient Boosting → AUC ≈ 0.94
* Random Forest → AUC ≈ 0.93
* XGBoost → AUC ≈ 0.93
*Decision Tree → AUC ≈ 0.90

* So in terms of ROC–AUC, the SVM is the best model.

5. MSE/MAE are regression metrics, they can be adapted for classification by comparing the predicted label (0/1) with the true label.
* [[91, 7],
 [ 6, 48]]
* Total = 91 + 7 + 6 + 48 = 152
* Misclassifications = 7 + 6 = 13
* Correct = 139
* MSE (Mean Squared Error)
-> Since predictions are 0/1, errors are either 0 or 1.
* MSE = (Number of errors/Total) = 13/152 ≈ 0.0855
* MAE (Mean Absolute Error)
* For 0/1 classification, MAE = MSE (same reasoning).
* MAE ≈ 0.0855

6. Best Model: SVM (RBF, tuned with GridSearchCV)
* Accuracy: 91.45% (Test)
* ROC–AUC: ~0.96 (highest among all models)
* MSE / MAE: ~0.085 (≈ 8.5% error rate)
* Confusion Matrix: TN=91, FP=7, FN=6, TP=48
* Class 1 (diabetic) Recall: 0.89 (very strong for medical use-cases)
