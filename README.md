# Blood Donation Prediction

A classification project that predicts whether a person will donate blood in a given month (March 2007), based on their past donation history. I tried seven different models on it and compared them side by side.

## Dataset

`blood_donation_data.csv` - 20,400 records, 4 features and 1 target.

| Column | What it means |
|---|---|
| Months since Last Donation | How long ago the person last donated |
| Number of Donations | Total donations made so far |
| Total Volume Donated (c.c.) | Total blood donated, in c.c. |
| Months since First Donation | How long ago they first donated |
| Made Donation in March 2007 | Target: 1 = donated, 0 = didn't |

The data follows the "Warm Up: Predict Blood Donations" problem format.

## What I found in the EDA

- The target is imbalanced: 12,240 people did not donate vs 8,160 who did.
- People who donated in March 2007 had given blood more recently (about 5 months since last donation vs about 10 for non-donors) and more often (about 5.9 donations vs 3.2 on average).
- Correlation with the target: Number of Donations (0.42) and Months since Last Donation (-0.38) matter most. Months since First Donation is basically unrelated (0.01).
- `Number of Donations` and `Total Volume Donated` are perfectly correlated (1.0), so they carry the same information.

## Approach

1. Checked the data (shape, dtypes, nulls, duplicates, correlations, group means) and plotted a heatmap, histogram and class distribution.
2. Split 80/20 (16,320 train / 4,080 test, `random_state=40`).
3. Scaled the features with `StandardScaler`.
4. Used **SMOTE** to balance the training set (9,776 samples in each class).
5. Trained and compared Logistic Regression, Naive Bayes, SVM, KNN (k picked with GridSearchCV), Decision Tree, Random Forest and Gradient Boosting.

Naive Bayes was trained on the original (non-SMOTE) data because I hit an error trying it on the resampled data.

## Results

Metrics are on the 4,080-row test set. Precision / Recall / F1 are for the "donated" class.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.743 | 0.656 | 0.738 | 0.695 |
| Naive Bayes | 0.730 | 0.727 | 0.511 | 0.600 |
| SVM (RBF, C=10) | 0.740 | 0.650 | 0.744 | 0.694 |
| KNN (k=11) | 0.722 | 0.630 | 0.720 | 0.672 |
| Decision Tree (depth 3) | 0.729 | 0.632 | 0.757 | 0.689 |
| Random Forest | 0.744 | 0.652 | 0.755 | 0.700 |
| Gradient Boosting | 0.744 | 0.659 | 0.735 | 0.695 |

All the models land within about 2 points of each other (72-74% accuracy). The fancier models didn't beat plain Logistic Regression by much, which tells me the limit here is the information in the 4 features, not the choice of algorithm. Naive Bayes had the best precision but missed about half of the actual donors.

## Tech used
- Python
- Pandas
- Scikit-learn (models, `GridSearchCV`, metrics, `StandardScaler`)
- imbalanced-learn (`SMOTE`)
- Seaborn, Matplotlib
- Jupyter Notebook

## Possible next steps
- Build a simple feature such as donations per month and see if it lifts the score
- Drop `Total Volume Donated` since it duplicates `Number of Donations`
