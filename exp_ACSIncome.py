import warnings
warnings.filterwarnings("ignore", message="Pandas requires version")
warnings.filterwarnings("ignore", message="A NumPy version >=")

# general imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# anonymity library
import pycanon
from anjana.anonymity import k_anonymity, l_diversity, t_closeness

# ML models
from xgboost import XGBClassifier as XGB

# our generic functions
from utils import get_metrics, write_suppression_results_to_csv, get_generalization_levels, get_train_test_data

# our data-specific functions
from utils import clean_process_adult_data, get_hierarchies_adult
from utils_ACSIncome import *
import config_experiments as cfg

#Import folktables dataset
import folktables
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSHealthInsurance, ACSPublicCoverage
from sklearn.metrics.pairwise import cosine_similarity



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




lst_supp_level = cfg.supp_level
dic_methods_parameters = {
        'k-anonymity': cfg.fixed_k,
        'l-diversity': cfg.fixed_l,
        't-closeness': cfg.fixed_t
        }

max_seed = cfg.max_seed
test_size = cfg.test_size


dataset = "ACSIncome"

#What is the k-anonimity level of the original ACSIncome
#define protected and sensitive attributes and Quasi-Identifiers
protected_att = 'SEX'
sens_att = 'PINCP'

#Applying k-anonymity to the new dataset


state = 'WY'




    
# Download and preprocess ACS data
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=[state], download=True)
features, target, _ = ACSIncome.df_to_pandas(acs_data)
entire_data = pd.concat([features, target.astype(int)], axis=1)


#Personalize the decision thresholds
lst_threshold_target = [int(acs_data['PINCP'].median()), int(acs_data['PINCP'].median()), acs_data['PINCP'].quantile(0.10)]

for method, anon_parameter in dic_methods_parameters.items():
    print(f"Method: {method}, Parameter: {anon_parameter}") 

    # Loop over several threshold targets (same dataset but with different Y distribution)
    for threshold_target in lst_threshold_target:
        print(f"Threshold target: {threshold_target}")

        # Setup the new target
        entire_data['PINCP'] = (acs_data['PINCP'] > threshold_target).astype(int)
        quasi_ident = list(set(entire_data.columns) - {protected_att} - {sens_att})
        SEED = 0
        max_seed = 50
        hierarchies = get_hierarchies_ACSIncome(entire_data)
        while SEED < max_seed:
            print(f"SEED: {SEED}")
            for supp_level in lst_supp_level:
                print(f"Suppression level : {supp_level}")
                try:
                    print(f"trying data anonymzation with k : {cfg.fixed_k}, suppression level {supp_level}")
                    # Split into train and test data
                    train_data, test_data = train_test_split(entire_data, test_size=test_size, random_state=SEED)
                    if 'index' in train_data.columns:
                        train_data.drop(['index']) 
                    train_data_anon = k_anonymity(train_data, [], quasi_ident, k=cfg.fixed_k, supp_level=supp_level, hierarchies=hierarchies)
                    #Remove the index col
                    if 'index' in train_data_anon.columns:
                        del train_data_anon['index'] 

                    # Assert that the level of k-anonymity is at least k
                    actual_k_anonymity = pycanon.anonymity.k_anonymity(train_data_anon, quasi_ident)
                    print(f"achieved level of k_anonymity : {actual_k_anonymity}")
                    assert actual_k_anonymity >= cfg.fixed_k, f"k-anonymity constraint not met: Expected >= {cfg.fixed_k}, but got {actual_k_anonymity}"

                    if method == 'l-diversity':
                        # Apply l-diversity
                        train_data_anon = l_diversity(train_data_anon, [], quasi_ident, sens_att, cfg.fixed_k, anon_parameter, supp_level, hierarchies)

                        # Assert that the level of l-diversity is exactly met
                        actual_l_diversity = pycanon.anonymity.l_diversity(train_data_anon, quasi_ident, [sens_att])
                        assert actual_l_diversity == anon_parameter, f"l-diversity constraint not met: Expected == {anon_parameter}, but got {actual_l_diversity}"
                        print(f"achieved level of l_diversity {actual_l_diversity}")
                    if method == 't-closeness':
                        # Apply t-closeness
                        train_data_anon = t_closeness(train_data_anon, [], quasi_ident, sens_att, cfg.fixed_k, anon_parameter, supp_level, hierarchies)
                        # Assert that the level of t-closeness is satisfied
                        actual_t_closeness = pycanon.anonymity.t_closeness(train_data_anon, quasi_ident, [sens_att], True)
                        assert actual_t_closeness <= anon_parameter, f"t-closeness constraint not met: Expected <= {anon_parameter}, but got {actual_t_closeness:.2f}"
                        print(f"achieved level of t_closeness : {actual_t_closeness}")

                    # Get generalization levels of the training set to apply the same to the test set
                    generalization_levels = get_generalization_levels(train_data_anon, quasi_ident, hierarchies)
                    train_data_anon = train_data_anon.dropna()
                    
                    
                    # Apply the same generalization levels to the test data (Except for the protected attribute: for fairness measurements)
                    for col in set(quasi_ident) - {protected_att}:
                        level = generalization_levels.get(col)
                        print(f"generalization level for col {col} : {level}")
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


                    #write_results_to_csv([SEED, dataset + "_" + str(threshold_target), protected_att, sens_att, method, anon_parameter, supp_level] + list(dic_metrics.values()), state, header=True)
                    write_suppression_results_to_csv([SEED, dataset + "_" + str(threshold_target), protected_att, sens_att, method, anon_parameter, supp_level] + list(dic_metrics.values()), state, header=True)
                    print("Results written in csv file")



                except Exception as e:
                    print(f"An error occurred for SEED {SEED}, k {cfg.fixed_k}: {e}")
                    continue

                SEED += 1
            print('-------------------------------------------------------------\n')
        print('=============================================================\n')
    print('#############################################################\n')
