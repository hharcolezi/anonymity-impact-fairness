# warning filters
import warnings
warnings.filterwarnings("ignore", message="Pandas requires version")
warnings.filterwarnings("ignore", message="A NumPy version >=")

# general imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# anonymity library
import pycanon
from anjana.anonymity import k_anonymity

# ML models
from xgboost import XGBClassifier as XGB

#Import folktables dataset
import folktables
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSHealthInsurance, ACSPublicCoverage
from sklearn.metrics.pairwise import cosine_similarity
# our generic functions
#from utils_ACSIncome import get_metrics, write_results_to_csv, get_generalization_levels, get_train_test_data

from utils_ACSIncome import *

# our data-specific functions
from utils import clean_process_adult_data, get_hierarchies_adult
import config_experiments as cfg












def individual_fairness(predictions, features, similarity_metric='euclidean', k=5):
    """
    Compute individual fairness for model predictions.

    Parameters:
        predictions (array-like): Model predictions for individuals.
        features (array-like): Feature matrix (N x D).
        similarity_metric (str): Similarity metric ('cosine' or 'euclidean').
        k (int): Number of nearest neighbors to consider.

    Returns:
        float: Average individual fairness score (lower is better).
    """
    # Compute pairwise similarity matrix
    if similarity_metric == 'cosine':
        similarity_matrix = cosine_similarity(features)
    elif similarity_metric == 'euclidean':
        similarity_matrix = -np.linalg.norm(features[:, None] - features, axis=2)
    else:
        raise ValueError("Unsupported similarity metric!")

    n = len(predictions)
    fairness_scores = []

    # For each individual, compare with k-nearest neighbors
    for i in range(n):
        # Get top k most similar individuals (excluding self)
        nearest_neighbors = np.argsort(-similarity_matrix[i])[:k + 1][1:]
        
        # Calculate prediction differences weighted by similarity
        for j in nearest_neighbors:
            fairness_scores.append(
                abs(predictions[i] - predictions[j]) * similarity_matrix[i, j]
            )

    # Return the average fairness score
    return np.mean(fairness_scores)





# Define the parameters
dataset = 'ACSIncome'
method = 'k-anonymity'

# Get parameters from config file
supp_level = cfg.supp_level[1]
lst_k = cfg.lst_k
max_seed = cfg.max_seed
test_size = cfg.test_size
if dataset == 'ACSIncome':
    lst_threshold_target = cfg.adult_threshold_target

state = 'WY'
# Main execution
write_results_to_csv([], state, header=True)
# Loop over several threshold targets (same dataset but with different Y distribution)
for threshold_target in lst_threshold_target:
    print(f"Threshold target: {threshold_target}")

    # read data
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state], download=True)
    features, target, _ = ACSIncome.df_to_pandas(acs_data)
    
    data = pd.concat([features, target.astype(int)], axis=1)
    data['PINCP'] = (acs_data['PINCP'] > threshold_target).astype(int)
    protected_att = 'SEX'
    sens_att = 'PINCP'  #target
    # Define the quasi-identifiers and the sensitive/protected attribute
    quasi_ident = list(set(data.columns) - {protected_att} - {sens_att})
    hierarchies = get_hierarchies_ACSIncome(data)
    # Loop over several seeds
    SEED = 0
    while SEED < max_seed:
        print(f"SEED: {SEED}")

        # Loop over several k values
        for k in lst_k:
            print(f"k: {k}")

            try:
                # Split into train and test data
                train_data, test_data = train_test_split(data, test_size=test_size, random_state=SEED)
                
                # Anonymize data
                train_data_anon = k_anonymity(train_data, [], quasi_ident, k, supp_level, hierarchies)
                if 'index' in train_data_anon.columns:
                    del train_data_anon['index'] 

                # Assert that the level of k-anonymity is at least k
                actual_k_anonymity = pycanon.anonymity.k_anonymity(train_data_anon, quasi_ident)
                print(f"achieved level of k_anonymity : {actual_k_anonymity}")
                assert actual_k_anonymity >= k, f"k-anonymity constraint not met: Expected >= {k}, but got {actual_k_anonymity}"

                if k > 1:
                    # Get generalization levels of the training set to apply the same to the test set
                    generalization_levels = get_generalization_levels(train_data_anon, quasi_ident, hierarchies)

                    # Apply the same generalization levels to the test data (Except for the protected attribute: for fairness measurements)
                    for col in set(quasi_ident) - {protected_att}:
                        level = generalization_levels.get(col)
                        
                        if level is not None:
                            # Retrieve the mapping dictionary for this level
                            hierarchy_mapping = dict(zip(hierarchies[col][0], hierarchies[col][level]))
                            
                            # Apply the mapping to the test data
                            test_data[col] = test_data[col].map(hierarchy_mapping)

                X_train, y_train, X_test, y_test = get_train_test_data(train_data_anon, test_data, sens_att)
                # Train the model
                model = XGB(enable_categorical=True, random_state=SEED, n_jobs=-1)
                model.fit(X_train, y_train)
                test_data = pd.concat([X_test, y_test], axis=1)
                df_fm = X_test.copy()
                df_fm[sens_att] = y_test.values
                y_pred_col = np.round(model.predict(X_test)).reshape(-1).astype(int)
                df_fm['y_pred'] = y_pred_col
                dic_metrics = get_metrics(df_fm, protected_att, sens_att)
                

                fairness_score = individual_fairness(y_pred_col, X_test.values, similarity_metric='euclidean', k=5)
                print(f"Individual Fairness Score: {fairness_score}")
                dic_metrics['INF'] = np.abs(fairness_score)
                print(f"metrics : {dic_metrics}")

                # Write results to csv
                write_results_to_csv([SEED, dataset + "_" + str(threshold_target), protected_att, sens_att, method, k, k] + list(dic_metrics.values()), state, header=False)

            except Exception as e:
                    print(f"An error occurred for SEED {SEED}, k {k}: {e}")
                    continue
        
        SEED += 1
        print('-------------------------------------------------------------\n')
    print('=============================================================\n')