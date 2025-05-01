import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np

finfraud_copy = pd.read_csv('~/New/finfraud_copy.csv')

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

# Training data is fyear = 1991-2001, testing data is fyear 2003 initially, then fyear 2004 but also expand training data to go up one year every time as well
for year in range(2003, 2009):
    X_train = X1_normalized_df[X1['fyear'] <= year - 2]
    X_test = X1_normalized_df[X1['fyear'] == year]
    y_train = y1[X1['fyear'] <= year - 2]
    y_test = y1[X1['fyear'] == year]

    # Create a Random Forest model
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict probabilities for the test set
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate AUC
    auc = roc_auc_score(y_test, y_proba)
    print("AUC for year {}: {}".format(year, auc))

    # Rank the instances based on predicted probabilities
    ranked_indices = np.argsort(y_proba)[::-1]  # Descending order

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
    y_pred_prob = model.predict_proba(X_test)

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

    # Calculate Area under the receiver operating characteristics (ROC) curve (AUC)
    auc = roc_auc_score(y_test, y_pred_prob[:,1])
    print(f'AUC for year {year}: {auc}')

    # Add results to DataFrame
    results = results.append({'year': year, 'auc': auc, 'accuracy': accuracy, 'NDCG_at_k': NDCG_at_k, 'precision': precision, 'precision_JAR': precision2, 'sensitivity': sensitivity, 'recall': recall, 'true_positives': tp, 'false_positives': fp, 'false_negatives': fn}, ignore_index=True)

print(results)
results.to_csv('~/New/RF/RF_results.csv', index=False)

