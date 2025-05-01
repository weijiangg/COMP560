import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, ndcg_score

def data_reader(data_path, year_start, year_end):
    # Read data from CSV file
    data = pd.read_csv(data_path)

    # Filter data based on years
    data = data[(data['fyear'] >= year_start) & (data['fyear'] <= year_end)]

    # Separate features and labels
    X = data.iloc[:, 4:]  # Assuming features start from column index 4
    y = data['misstate']

    return X, y

# Set parameters
start_year = 1991
end_year = 2014

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Year_Test', 'AUC', 'NDCG@1%', 'NDCG@2%', 'NDCG@3%', 'NDCG@4%', 'NDCG@5%'])

# Loop through each testing year
for year_test in range(2003, 2015):
    print(f"==> Running XGBoost (training period: {start_year}-{year_test-2}, testing period: {year_test}, with 2-year gap)...")
    
    # Read training data
    X_train, y_train = data_reader('~/data_FraudDetection_JAR2020.csv', start_year, year_test-2)

    # Read testing data
    X_test, y_test = data_reader('~/data_FraudDetection_JAR2020.csv', year_test, year_test)

    # Define the XGBoost model
    xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, min_child_weight=5)

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_proba)

    # Calculate NDCG@k
    k_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    ndcg_scores = []
    for k in k_values:
        k_top = int(len(y_test) * k)
        ndcg = ndcg_score([y_test], [y_pred_proba], k=k_top)
        ndcg_scores.append(ndcg)

    # Store results in DataFrame
    results_df = results_df.append({'Year_Test': year_test,
                                    'AUC': auc,
                                    'NDCG@1%': ndcg_scores[0],
                                    'NDCG@2%': ndcg_scores[1],
                                    'NDCG@3%': ndcg_scores[2],
                                    'NDCG@4%': ndcg_scores[3],
                                    'NDCG@5%': ndcg_scores[4]}, ignore_index=True)

# Write results to CSV
results_df.to_csv('results.csv', index=False)
