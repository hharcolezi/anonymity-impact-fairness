# warning filters
import warnings
warnings.filterwarnings("ignore", message="Pandas requires version")
warnings.filterwarnings("ignore", message="A NumPy version >=")

# general imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
# anonymity library
import pycanon
from anjana.anonymity import k_anonymity, l_diversity, t_closeness

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



def estimate_lipschitz_constant(predictions, features, similarity_metric='euclidean'):
    """
    Estimate the Lipschitz constant for a classifier based on predictions and feature similarities.

    Parameters:
        predictions (array-like): Model predictions for individuals (N x 1).
        features (array-like): Feature matrix (N x D).
        similarity_metric (str): Similarity metric ('cosine' or 'euclidean').

    Returns:
        float: Estimated Lipschitz constant (larger values indicate higher violation of individual fairness).
    """
    # Compute pairwise similarity matrix
    if similarity_metric == 'cosine':
        similarity_matrix = cosine_similarity(features)
    elif similarity_metric == 'euclidean':
        similarity_matrix = np.linalg.norm(features[:, None] - features, axis=2)
    else:
        raise ValueError("Unsupported similarity metric!")

    n = len(predictions)
    lipschitz_ratios = []

    # Iterate over each pair of individuals to compute the Lipschitz ratio
    for i in range(n):
        for j in range(i + 1, n):  # Avoid redundant calculations (i < j)
            # Calculate the distance between the feature vectors
            distance = similarity_matrix[i, j]
            #print(distance)
            if distance == 0:  # Skip pairs with identical features (no change in prediction)
                continue

            # Calculate the absolute difference in predictions
            prediction_difference = entropy(predictions[i],predictions[j]) #KL-Divergence
            # Estimate the Lipschitz ratio for the pair
            lipschitz_ratio = prediction_difference / distance
            lipschitz_ratios.append(lipschitz_ratio)

    # Return the maximum Lipschitz ratio as the estimated Lipschitz constant
    return np.max(lipschitz_ratios) if lipschitz_ratios else 0.0


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
methods = ['k-anonymity', 'l-diversity', 't-closeness']
model_types = ['XGBoost', 'DNN'] #Supported model types
model_type = model_types[0]

method = methods[0]

t_params = [0.45, 0.5, 0.55]
l_params = [2]
count_l = 0
count_t = 0
# Get parameters from config file
supp_level = cfg.supp_level[1]
lst_k = cfg.lst_k
max_seed = cfg.max_seed
max_seed = 20
test_size = cfg.test_size

state = 'ALL'
fraction = 0.10

# read data (and potentially download)
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
if state != 'ALL' : 
    acs_data = data_source.get_data(states=[state], download=True)
    features, target, _ = ACSIncome.df_to_pandas(acs_data)
elif state == 'ALL' : 
    all_states = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    ]
    all_features = []
    all_targets = []  
    for curr_state in all_states[:2] : 
        print(f'extracting state {curr_state}')
        acs_data = data_source.get_data(states = [curr_state], download=False)
        features, target, _ = ACSIncome.df_to_pandas(acs_data)

        state_features = features.sample(n=500, random_state=42)
        #state_target = target.loc[state_features.index] 
        state_targets = acs_data['PINCP'].loc[state_features.index]

        all_features.append(state_features)  # Append the features to the list
        all_targets.append(state_targets) 
        

    features = pd.concat(all_features, axis=0, ignore_index=True)
    targets = pd.concat(all_targets, axis=0, ignore_index=True)

    print('dataset extracted')



if state != 'ALL' : 
    data = pd.concat([features, targets.astype(int)], axis=1)
    #Personalize the decision thresholds
    lst_threshold_target = [int(acs_data['PINCP'].quantile(0.25)), int(acs_data['PINCP'].median()), int(acs_data['PINCP'].quantile(0.75))]


elif state == 'ALL' : 
    data = pd.concat([features, targets], axis=1)
    lst_threshold_target = [int(data['PINCP'].quantile(0.25)), int(data['PINCP'].median()), int(data['PINCP'].quantile(0.75))]
    # Shuffle the dataset rows
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)




# Main execution (create the file and write the headers)
write_results_to_csv([], state, model_type, header=True)
# Loop over several threshold targets (same dataset but with different Y distribution)

for threshold_target in lst_threshold_target:
    print(f"Threshold target: {threshold_target}")
    #Setup the new target only if we are working with a single state
    if state != 'ALL' : 
        print('setting up the threshold')
        data['PINCP'] = (acs_data['PINCP'] > threshold_target).astype(int)
    elif state == 'ALL' :
        print('setting up the threshold')
        data['PINCP'] = (data['PINCP'] > threshold_target).astype(int)
    
    protected_att = 'SEX'
    sens_att = 'PINCP'  #target
    # Define the quasi-identifiers and the sensitive and protected attribute
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
                anon_parameter = k
                # Anonymize the training data
                if k > 1 :
                    train_data_anon = k_anonymity(train_data, [], quasi_ident, k, supp_level, hierarchies)
                elif k == 1 : 
                    train_data_anon = train_data.copy()
                if 'index' in train_data_anon.columns:
                    del train_data_anon['index'] 
                # Assert that the level of k-anonymity is at least k
                actual_k_anonymity = pycanon.anonymity.k_anonymity(train_data_anon, quasi_ident)
                print(f"achieved level of k_anonymity : {actual_k_anonymity}")
                assert actual_k_anonymity >= k, f"k-anonymity constraint not met: Expected >= {k}, but got {actual_k_anonymity}"

                if method == 'l-diversity':
                    # Apply l-diversity
                    anon_parameter = 2
                    if k > 1 : 
                        train_data_anon = l_diversity(train_data_anon, [], quasi_ident, sens_att, cfg.fixed_k, anon_parameter, supp_level, hierarchies)
                    elif k == 1 : 
                        train_data_anon = train_data.copy()
                    # Assert that the level of l-diversity is exactly met
                    actual_l_diversity = pycanon.anonymity.l_diversity(train_data_anon, quasi_ident, [sens_att])
                    assert actual_l_diversity == anon_parameter, f"l-diversity constraint not met: Expected == {anon_parameter}, but got {actual_l_diversity}"
                    print(f"achieved level of l_diversity {actual_l_diversity}")
                
                if method == 't-closeness':
                    # Apply t-closeness
                    anon_parameter = t_params[count_t % len(t_params)]
                    if k > 1 :
                        train_data_anon = t_closeness(train_data_anon, [], quasi_ident, sens_att, cfg.fixed_k, anon_parameter, supp_level, hierarchies)
                    elif k == 1 : 
                        train_data_anon = train_data.copy()
                    # Assert that the level of t-closeness is satisfied
                    actual_t_closeness = pycanon.anonymity.t_closeness(train_data_anon, quasi_ident, [sens_att], True)
                    assert actual_t_closeness <= anon_parameter, f"t-closeness constraint not met: Expected <= {anon_parameter}, but got {actual_t_closeness:.2f}"
                    count_t+=1  #move to the next target value
                    print(f"achieved level of t_closeness : {actual_t_closeness}")

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
                #Train a model (XGBoost or DNN)
                if model_type == 'XGBoost' : 
                    model = XGB(enable_categorical=True, random_state=SEED, n_jobs=-1)
                    model.fit(X_train, y_train)
                    
                elif model_type == 'DNN' : 
                    model = folktables_DNN((X_train.shape[1],), init_distrib='lecun_uniform')
                    model.fit(X_train, y_train, epochs=75, verbose=0)
                    print('Achieved test accuracy : ', model.evaluate(X_test, y_test, verbose=0)[1])
                test_data = pd.concat([X_test, y_test], axis=1)
                df_fm = X_test.copy()
                df_fm[sens_att] = y_test.values
                #This one will be used for individual fairness
                soft_ypred = model.predict_proba(X_test)
                #This one will be used for group fairness
                y_pred_col = model.predict(X_test)
                df_fm['y_pred'] = y_pred_col
                dic_metrics = get_metrics(df_fm, protected_att, sens_att)
                
                #Add NCP to the measures + Sample 10%-20%-... from all US dataset
                fairness_score = estimate_lipschitz_constant(soft_ypred, X_test.values, similarity_metric='euclidean')
                #fairness_score = individual_fairness(y_pred_col, X_test.values, similarity_metric='euclidean', k=5)
                ncp = calculate_total_ncp(train_data, train_data_anon)
                print(f"Individual Fairness Score: {fairness_score}")
                dic_metrics['INF'] = np.abs(fairness_score)
                dic_metrics['NCP'] = ncp
                print(f"metrics : {dic_metrics}")

                # Write results to csv
                write_results_to_csv([SEED, dataset + "_" + str(threshold_target), protected_att, sens_att, method, k, anon_parameter] + list(dic_metrics.values()), state, model_type, header=False)
                print('results written')
            except Exception as e:
                    print(f"An error occurred for SEED {SEED}, k {k}: {e}")
                    continue
        
        SEED += 1
        method = methods[SEED%3] #Apply the next method
        print('-------------------------------------------------------------\n')
    print('=============================================================\n')
