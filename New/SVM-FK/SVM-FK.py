import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

# Load the data
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

# Define the financial kernel
def financial_kernel(X1, X2):
    n_features = X1.shape[1] // 2  # Assuming each feature has two years of data
    kernel = np.zeros((X1.shape[0], X2.shape[0]))

    for i in range(n_features):
        for j in range(i + 1, n_features):
            A1 = X1[:, i]
            A2 = X1[:, i + n_features]
            L1 = X1[:, j]
            L2 = X1[:, j + n_features]

            B1 = X2[:, i]
            B2 = X2[:, i + n_features]
            K1 = X2[:, j]
            K2 = X2[:, j + n_features]

            kernel += (A1 / L1)[:, None] * (B1 / K1)[None, :]
            kernel += (L1 / A1)[:, None] * (K1 / B1)[None, :]
            kernel += (L2 / A2)[:, None] * (K2 / B2)[None, :]
            kernel += (A2 / L2)[:, None] * (B2 / K2)[None, :]
            kernel += (A1 * L2 / (A2 * L1))[:, None] * (B1 * K2 / (B2 * K1))[None, :]
            kernel += (L1 * A2 / (L2 * A1))[:, None] * (K1 * B2 / (K2 * B1))[None, :]

    # deal with NaN values or zero division or infinity with median 
    kernel = np.nan_to_num(kernel, nan=np.nanmedian(kernel), posinf=np.nanmedian(kernel), neginf=np.nanmedian(kernel))
    
    return kernel

# Training data is fyear = 1991-2001, testing data is fyear 2003 initially, then fyear 2004 but also expand training data to go up one year every time as well
for year in range(2003, 2009):
    X_train2 = X1_normalized_df[X1['fyear'] <= year - 2].values
    X_test = X1_normalized_df[X1['fyear'] == year].values
    y_train2 = y1[X1['fyear'] <= year - 2]
    y_test = y1[X1['fyear'] == year]

    # print number of misstate = 1 and 0 in X_train2
    print(y_train2.value_counts())

    # pick same number of fraud and non-fraud cases for training and validation not using random 
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    X_train, y_train = rus.fit_resample(X_train2, y_train2)

    # display x train and y train shape
    print(X_train.shape, y_train.shape)

    # Create an SVM model with a financial kernel
    model = SVC(kernel=financial_kernel, probability=True, random_state=42)

    # without kernel
    model = SVC(probability=True, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict probabilities for the test set
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate AUC
    auc = roc_auc_score(y_test, y_proba)
    print("AUC for year {}: {}".format(year, auc))

    # Predict probabilities for the test set
    y_proba_test = model.predict_proba(X_test)[:, 1]

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

    # Calculate Area under the receiver operating characteristics (ROC) curve (AUC).
    auc = roc_auc_score(y_test, y_pred_prob[:,1])
    print(f'AUC for year {year}: {auc}')

    # Add results to DataFrame
    results = results.append({'year': year, 'auc': auc, 'accuracy': accuracy, 'NDCG_at_k' : NDCG_at_k , 'precision': precision, 'precision_JAR': precision2, 'sensitivity': sensitivity, 'recall': recall, 'true_positives': tp, 'false_positives': fp, 'false_negatives': fn}, ignore_index=True)

print(results)

# Save the results to a CSV file
results.to_csv('~/New/SVM-FK/SVM_FK_results.csv', index=False)
