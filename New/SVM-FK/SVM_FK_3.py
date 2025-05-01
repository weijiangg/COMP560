import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

# Load the data
finfraud_copy = pd.read_csv('~/New/finfraud_copy.csv')

# Keep only the first 28 columns
finfraud_copy = finfraud_copy.iloc[:, :31]

# Initialize the results DataFrame
results = pd.DataFrame(columns=['year', 'auc', 'accuracy', 'precision', 'recall', 'true_positives', 'false_positives', 'false_negatives'])

# Assume 'finfraud_copy' is your DataFrame
X1 = finfraud_copy.drop(['misstate', 'p_aaer', 'gvkey'], axis=1)
y1 = finfraud_copy['misstate']

# Normalize the values in X1
scaler = StandardScaler()
X1_normalized = scaler.fit_transform(X1)

# Adding the normalized features back into the DataFrame for easy indexing
X1_normalized_df = pd.DataFrame(X1_normalized, columns=X1.columns)

def compute_financial_ratios(df):
    ratios = []
    n = df.shape[1] // 2  # Assumes half the columns are for year 1, half for year 2
    for i in range(n):
        for j in range(i + 1, n):
            ratios.append(df.iloc[:, i] / df.iloc[:, j])
            ratios.append(df.iloc[:, j] / df.iloc[:, i])
            ratios.append(df.iloc[:, i + n] / df.iloc[:, j + n])
            ratios.append(df.iloc[:, j + n] / df.iloc[:, i + n])
            ratios.append((df.iloc[:, i] * df.iloc[:, j + n]) / (df.iloc[:, j] * df.iloc[:, i + n]))
            ratios.append((df.iloc[:, j] * df.iloc[:, i + n]) / (df.iloc[:, i] * df.iloc[:, j + n]))
    return pd.concat(ratios, axis=1)

def financial_kernel(X, Y):
    ratios_X = compute_financial_ratios(pd.DataFrame(X))
    ratios_Y = compute_financial_ratios(pd.DataFrame(Y))
    return np.dot(ratios_X, ratios_Y.T)

# Training data is fyear = 1991-2001, testing data is fyear 2003 initially, then fyear 2004 but also expand training data to go up one year every time as well
for year in range(2003, 2009):
    X_train2 = X1_normalized_df[X1['fyear'] <= year - 2].values
    X_test = X1_normalized_df[X1['fyear'] == year].values
    y_train2 = y1[X1['fyear'] <= year - 2]
    y_test = y1[X1['fyear'] == year]

    # Pick same number of fraud and non-fraud cases for training and validation not using random 
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    X_train, y_train = rus.fit_resample(X_train2, y_train2)

    # Display X_train and y_train shape
    print(X_train.shape, y_train.shape)

    # Cost-sensitive SVM with grid search for C+1:C-1 ratio
    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'class_weight': [{1: 20, 0: 1}]}
    model = SVC(kernel=financial_kernel, probability=True, random_state=10)
    grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Fit the best model
    best_model.fit(X_train, y_train)

    # Predict probabilities for the test set
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculate AUC
    auc = roc_auc_score(y_test, y_proba)
    print("AUC for year {}: {}".format(year, auc))

    # Predict probabilities for the test set
    y_proba_test = best_model.predict_proba(X_test)[:, 1]

    # Rank the instances based on predicted probabilities
    ranked_indices = np.argsort(y_proba_test)[::-1]  # Descending order

    # Define k (e.g., top 1%)
    k = int(len(y_test) * 0.01)

    # Select the top k instances
    top_k_indices = ranked_indices[:k]

    # Calculate true positives (TP) in the top 1%
    TP = y_test.iloc[top_k_indices].sum()

    # Calculate false positives (FP) in the top 1%
    FP = k - TP

    # Calculate false negatives (FN) in the bottom 99%
    FN = y_test.iloc[ranked_indices[k:]].sum()

    # Calculate sensitivity
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    print("Sensitivity for year {}: {}".format(year, sensitivity))

    # Calculate DCG@k
    DCG_at_k = sum((2 ** y_test.iloc[ranked_indices[i]] - 1) / np.log2(i + 2) for i in range(k))

    # Ideal DCG@k (when all true frauds are ranked at the top)
    ideal_DCG_at_k = sum((2 ** 1 - 1) / np.log2(i + 2) for i in range(k)) 

    # Calculate NDCG@k
    NDCG_at_k = DCG_at_k / ideal_DCG_at_k if ideal_DCG_at_k != 0 else 0
    print("NDCG@{} for year {}: {}".format(k, year, NDCG_at_k))

    # Calculate Precision
    precision2 = TP / (TP + FP)
    print("Precision for year {}: {}".format(year, precision2))

    # Make predictions with probabilities
    y_pred_prob = best_model.predict_proba(X_test)

    # Find optimal threshold based on precision and recall
    thresholds = np.arange(0, 1, 0.001)
    scores = [precision_score(y_test, (y_pred_prob[:,1] > threshold).astype(int)) for threshold in thresholds]
    best_threshold = thresholds[scores.index(max(scores))]
    print(f'Best threshold for year {year}: {best_threshold}')

    # Convert probabilities to binary predictions based on the optimal threshold
    y_pred = (y_pred_prob[:,1] > best_threshold).astype(int)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for year {year}: {accuracy}')

    # Calculate precision and recall
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f'Precision for year {year}: {precision}')
    print(f'Recall for year {year}: {recall}')

    # Calculate true positives, false positives, and false negatives
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f'True Positives for year {year}: {tp}')
    print(f'False Positives for year {year}: {fp}')
    print(f'False Negatives for year {year}: {fn}')

    # Calculate Area under the receiver operating characteristics (ROC) curve (AUC).
    auc = roc_auc_score(y_test, y_pred_prob[:,1])
    print(f'AUC for year {year}: {auc}')

    # Add results to DataFrame
    results = results.append({'year': year, 'auc': auc, 'accuracy': accuracy, 'NDCG_at_k': NDCG_at_k, 'precision': precision, 'precision_JAR': precision2, 'sensitivity': sensitivity, 'recall': recall, 'true_positives': tp, 'false_positives': fp, 'false_negatives': fn}, ignore_index=True)

print(results)

# Save the results to a CSV file
results.to_csv('~/New/SVM-FK/SVM_FK3_results.csv', index=False)
