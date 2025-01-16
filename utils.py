
import typing
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import csv
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import math
import ray
import os
from sklearn.neighbors import NearestNeighbors


def get_metrics(df_fm, protected_attribute, target):
    
    dic_metrics = {# Statistical Parity Difference
                "SPD": np.nan, 

                # Equal Opportunity Difference
                "EOD": np.nan, 

                # Model Accuracy Difference
                "MAD": np.nan, 

                # Predictive Equality Disparity
                "PED": np.nan,

                # Predictive Rate Disparity
                "PRD": np.nan,

                # Accuracy Score
                "ACC": np.nan,

                # f1 Score
                "f1": np.nan, 

                # Precision Score
                "Precision": np.nan,

                # Recall Score
                "Recall": np.nan, 

                # ROC AUC Score
                "ROC_AUC": np.nan,

                # Confusion Matrix
                "CM": np.nan,
                }
    
    # Filtering datasets for fairness metrics
    df_a_1 = df_fm.loc[df_fm[protected_attribute] == 1]
    df_a_0 = df_fm.loc[df_fm[protected_attribute] == 0]

    # Calculate Statistical Parity per group
    SP_a_1 = df_a_1.loc[df_a_1["y_pred"] == 1].shape[0] / df_a_1.shape[0]
    SP_a_0 = df_a_0.loc[df_a_0["y_pred"] == 1].shape[0] / df_a_0.shape[0]
    
    # Statistical Parity Difference
    SPD = SP_a_1 - SP_a_0
    dic_metrics["SPD"] = SPD

    # Equal Opportunity
    EO_a_1 = recall_score(df_a_1[target], df_a_1['y_pred'])
    EO_a_0 = recall_score(df_a_0[target], df_a_0['y_pred'])

    # Equal Opportunity Difference
    EOD = EO_a_1 - EO_a_0
    dic_metrics["EOD"] = EOD

    # Mode Accuracy
    MA_a_1 = accuracy_score(df_a_1[target], df_a_1['y_pred'])
    MA_a_0 = accuracy_score(df_a_0[target], df_a_0['y_pred'])

    # Model Accuracy Difference
    MAD = MA_a_1 - MA_a_0
    dic_metrics["MAD"] = MAD

    # Predictive Equality Disparity (False Positive Rate difference)
    FPR_a_1 = (df_a_1[(df_a_1["y_pred"] == 1) & (df_a_1[target] == 0)].shape[0]) / (df_a_1[target] == 0).sum()
    FPR_a_0 = (df_a_0[(df_a_0["y_pred"] == 1) & (df_a_0[target] == 0)].shape[0]) / (df_a_0[target] == 0).sum()
    PED = FPR_a_1 - FPR_a_0
    dic_metrics["PED"] = PED

    # Predictive Rate Disparity (Positive Predictive Value difference)
    PPV_a_1 = precision_score(df_a_1[target], df_a_1['y_pred'])
    PPV_a_0 = precision_score(df_a_0[target], df_a_0['y_pred'])
    PRD = PPV_a_1 - PPV_a_0
    dic_metrics["PRD"] = PRD

    # Accuracy Score
    dic_metrics["ACC"] = accuracy_score(df_fm[target], df_fm['y_pred'])
    
    # f1 Score
    dic_metrics["f1"] = f1_score(df_fm[target], df_fm['y_pred'])

    # Precision Score
    dic_metrics["Precision"] = precision_score(df_fm[target], df_fm['y_pred'])

    # Recall Score
    dic_metrics["Recall"] = recall_score(df_fm[target], df_fm['y_pred'])

    # ROC AUC Score
    dic_metrics["ROC_AUC"] = roc_auc_score(df_fm[target], df_fm['y_pred'])

    # Confusion Matrix
    dic_metrics["CM"] = confusion_matrix(df_fm[target], df_fm['y_pred'])
    
    return dic_metrics

def write_main_results_to_csv(values, dataset, header=False):
    """Write the results to a csv file."""

    file_path = "results/anonymity_impact_fairness_" + dataset + ".csv"
    # Check if the file exists and is empty
    file_exists = os.path.isfile(file_path)
    file_empty = os.stat(file_path).st_size == 0 if file_exists else True

    with open(file_path, mode='a', newline='') as scores_file:
        scores_writer = csv.writer(scores_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header and file_empty:# Write header if specified and file is empty
            scores_writer.writerow(["SEED", "protected_att", "target", "method", "k_parameter", "anon_parameter", "SPD", "EOD", "MAD", "PED", "PRD", "ACC", "f1", "Precision", "Recall", "ROC_AUC", "CM", "ASF", "ALF", "NCF"])
        if not header: # Write the actual values
            scores_writer.writerow(values)

def write_target_distribution_results_to_csv(values, dataset, header=False):
    """Write the results to a csv file."""

    file_path = "results/anonymity_impact_fairness_target_distribution_" + dataset + ".csv"
    # Check if the file exists and is empty
    file_exists = os.path.isfile(file_path)
    file_empty = os.stat(file_path).st_size == 0 if file_exists else True

    with open(file_path, mode='a', newline='') as scores_file:
        scores_writer = csv.writer(scores_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header and file_empty:# Write header if specified and file is empty
            scores_writer.writerow(["SEED", "protected_att", "target", "method", "anon_parameter", "threshold_target", "SPD", "EOD", "MAD", "PED", "PRD", "ACC", "f1", "Precision", "Recall", "ROC_AUC", "CM", "ASF", "ALF", "NCF"])
        if not header: # Write the actual values
            scores_writer.writerow(values)

def write_suppression_results_to_csv(values, dataset, header=False):
    """Write the results to a csv file."""

    file_path = "results/anonymity_impact_fairness_suppression_" + dataset + ".csv"
    # Check if the file exists and is empty
    file_exists = os.path.isfile(file_path)
    file_empty = os.stat(file_path).st_size == 0 if file_exists else True

    with open(file_path, mode='a', newline='') as scores_file:
        scores_writer = csv.writer(scores_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header and file_empty:# Write header if specified and file is empty
            scores_writer.writerow(["SEED", "protected_att", "target", "method", "anon_parameter", "supp_level", "SPD", "EOD", "MAD", "PED", "PRD", "ACC", "f1", "Precision", "Recall", "ROC_AUC", "CM", "ASF", "ALF", "NCF"])
        if not header: # Write the actual values
            scores_writer.writerow(values)

def write_classifier_results_to_csv(values, dataset, header=False):
    """Write the results to a csv file."""

    file_path = "results/anonymity_impact_fairness_classifier_" + dataset + ".csv"
    # Check if the file exists and is empty
    file_exists = os.path.isfile(file_path)
    file_empty = os.stat(file_path).st_size == 0 if file_exists else True

    with open(file_path, mode='a', newline='') as scores_file:
        scores_writer = csv.writer(scores_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header and file_empty:# Write header if specified and file is empty
            scores_writer.writerow(["SEED", "protected_att", "target", "model_name", "method", "anon_parameter", "SPD", "EOD", "MAD", "PED", "PRD", "ACC", "f1", "Precision", "Recall", "ROC_AUC", "CM", "ASF", "ALF", "NCF"])
        if not header: # Write the actual values
            scores_writer.writerow(values)
           
def write_data_fraction_results_to_csv(values, dataset, header=False):
    """Write the results to a csv file."""

    file_path = "results/anonymity_impact_fairness_data_fraction_" + dataset + ".csv"
    # Check if the file exists and is empty
    file_exists = os.path.isfile(file_path)
    file_empty = os.stat(file_path).st_size == 0 if file_exists else True

    with open(file_path, mode='a', newline='') as scores_file:
        scores_writer = csv.writer(scores_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header and file_empty:# Write header if specified and file is empty
            scores_writer.writerow(["SEED", "protected_att", "target", "method", "anon_parameter", "fraction", "SPD", "EOD", "MAD", "PED", "PRD", "ACC", "f1", "Precision", "Recall", "ROC_AUC", "CM", "ASF", "ALF", "NCF"])
        if not header: # Write the actual values
            scores_writer.writerow(values)
 
def clean_process_data(data, dataset, sens_att, protected_att, threshold_target=None):
    """Clean and preprocess the dataset."""
    
    if dataset == 'adult':
        data.columns = data.columns.str.strip()
        data.drop(columns=['capital-gain', 'capital-loss', 'education-num'], inplace=True)
        cat_cols = [ "workclass", "education", "marital-status", "occupation", "relationship",  "gender", "native-country", "race"]
        for col in cat_cols:
            data[col] = data[col].str.strip()

        # drop nans
        data = data.replace({'native-country': {'?': np.nan}, 'workclass': {'?': np.nan}, 'occupation': {'?': np.nan}})
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Transform protected and target attributes
        data[sens_att] = data[sens_att].apply(lambda x: int(x>threshold_target))
        if protected_att == 'gender':
            data[protected_att] = data[protected_att].apply(lambda x: 1 if x == 'Male' else 0)
        elif protected_att == 'race':
            data[protected_att] = data[protected_att].apply(lambda x: 1 if x == 'White' else 0)

    elif dataset == 'bank':
        data.columns = data.columns.str.strip()
        cat_cols = [ "job", "marital", "education", "default", "housing",  "loan", "contact", "month", "poutcome"]
        for col in cat_cols:
            data[col] = data[col].str.strip()

        # Transform protected and target attributes
        data[sens_att] = data[sens_att].apply(lambda x: int(x=='yes'))
        if protected_att == 'age':
            data[protected_att] = data[protected_att].apply(lambda x: 1 if 25 <= x <= 60 else 0)
        elif protected_att == 'marital':
            data[protected_att] = data[protected_att].apply(lambda x: 1 if x == 'married' else 0)

        data.reset_index(drop=True, inplace=True)   


    elif dataset == 'ACSIncome':
        data.columns = data.columns.str.strip()

        # Transform protected and target attributes
        data[sens_att] = data[sens_att].apply(lambda x: int(x>threshold_target))
        if protected_att == 'SEX':
            data[protected_att] = data[protected_att].apply(lambda x: 1 if x == 1 else 0)
        elif protected_att == 'RAC1P':
            data[protected_att] = data[protected_att].apply(lambda x: 1 if x == 1 else 0)

        data.reset_index(drop=True, inplace=True)   


    return data

def get_hierarchies(data, dataset):
    """Generate/read hierarchies for quasi-identifiers based on provided data."""

    if dataset == 'adult':
        return {
                "age": dict(pd.read_csv("hierarchies/adult/age.csv", header=None)),
                "native-country": dict(pd.read_csv("hierarchies/adult/country.csv", header=None)),
                "education": dict(pd.read_csv("hierarchies/adult/education.csv", header=None)),
                "marital-status": dict(pd.read_csv("hierarchies/adult/marital.csv", header=None)),
                "occupation": dict(pd.read_csv("hierarchies/adult/occupation.csv", header=None)),
                "race": dict(pd.read_csv("hierarchies/adult/race.csv", header=None)),
                "workclass": dict(pd.read_csv("hierarchies/adult/workclass.csv", header=None)),
                "hours-per-week": {0: pd.Series(range(data["hours-per-week"].max()+1)),
                                1: generate_intervals(range(data["hours-per-week"].max()+1), 0, 100, 5),
                                2: generate_intervals(range(data["hours-per-week"].max()+1), 0, 100, 25),
                                3: generate_intervals(range(data["hours-per-week"].max()+1), 0, 100, 50),
                                4: np.array(["*"] * int(data["hours-per-week"].max()+1))},
                "relationship": {0: data["relationship"].unique(),
                                1: np.array(["*"] * len(data["relationship"].unique()))},
                "gender": {0: data["gender"].unique(),
                        1: np.array(["*"] * len(data["gender"].unique()))}  
                }
    

    elif dataset == 'ACSIncome':
        return {
                "AGEP": dict(pd.read_csv("hierarchies/ACSIncome/AGEP.csv", header=None)),
                "COW" : dict(pd.read_csv("hierarchies/ACSIncome/COW.csv", header=None)),
                "SCHL": dict(pd.read_csv("hierarchies/ACSIncome/SCHL.csv", header=None)),
                "MAR" : dict(pd.read_csv("hierarchies/ACSIncome/MAR.csv", header=None)),
                "OCCP" : dict(pd.read_csv("hierarchies/ACSIncome/OCCP.csv", header=None)),
                "POBP" : dict(pd.read_csv("hierarchies/ACSIncome/POBP.csv", header=None)),
                "WAOB" : dict(pd.read_csv("hierarchies/ACSIncome/WAOB.csv", header=None)),
                "RELP" : dict(pd.read_csv("hierarchies/ACSIncome/RELP.csv", header=None)),
                "RAC1P": dict(pd.read_csv("hierarchies/ACSIncome/RAC1P.csv", header=None)),
                "PINCP": dict(pd.read_csv("hierarchies/ACSIncome/PINCP.csv", header=None)),
                "WKHP": {0: pd.Series(range(int(data["WKHP"].max()) + 1)),  # Get the max value and convert to int
                        1: generate_intervals(range(int(data["WKHP"].max()) + 1), 0, 100, 5),
                        2: generate_intervals(range(int(data["WKHP"].max()) + 1), 0, 100, 25),
                        3: generate_intervals(range(int(data["WKHP"].max()) + 1), 0, 100, 50),
                        4: np.array(["*"] * (int(data["WKHP"].max()) + 1))  # Convert max value to int
                        },
                "SEX": {0: data["SEX"].unique(),
                        1: np.array(["*"] * len(data["SEX"].unique()))}  
                }

def get_generalization_levels(train_data_anon, quasi_ident, hierarchies):
    """Get the generalization levels of the training set to apply the same to the test set."""

    generalization_levels = {
        col: next(
            (
                level
                for level, values in hierarchies[col].items()
                if set(train_data_anon[col].unique()).issubset(
                    set(v for v in values if not pd.isna(v))  # Filter out NaN values
                )
            ),
            None,
        )
        for col in quasi_ident
    }

    return generalization_levels

def get_train_test_data(train_data_anon, test_data, sens_att):
    """Get the train and test data for the model training."""

    X_train, y_train = train_data_anon.drop(columns=sens_att), train_data_anon[sens_att]
    X_test, y_test = test_data.drop(columns=sens_att), test_data[sens_att]

    # Label encode based on concatenated train and test data
    X_combined = pd.concat([X_train, X_test], axis=0)
    label_encoders = {}
    for col in X_combined.columns:
        le = LabelEncoder()
        X_combined[col] = le.fit_transform(X_combined[col].astype(str))
        label_encoders[col] = le

    # Split the encoded combined data back into X_train and X_test
    X_train = X_combined.iloc[:len(X_train), :].reset_index(drop=True)
    X_test = X_combined.iloc[len(X_train):, :].reset_index(drop=True)

    return X_train, y_train, X_test, y_test

def generate_intervals(
                        quasi_ident: typing.Union[typing.List, np.ndarray],
                        inf: typing.Union[int, float],
                        sup: typing.Union[int, float],
                        step: int,
                    ) -> pd.Series:
    """
    Fixing the function to generate intervals as hierarchies from <https://github.com/IFCA-Advanced-Computing/anjana/blob/main/anjana/anonymity/utils/utils.py>.
    Generate intervals as hierarchies.

    Given a quasi-identifier of numeric type, creates a pandas Series containing
    interval-based generalizations (hierarchies) for each value of the quasi-identifier.
    The intervals will have the length specified by the 'step' parameter.

    :param quasi_ident: values of the quasi-identifier to be generalized
    :type quasi_ident: list or numpy array

    :param inf: lower bound for the set of intervals
    :type inf: int or float

    :param sup: upper bound for the set of intervals
    :type sup: int or float

    :param step: size of each interval
    :type step: int

    :return: pandas Series with the intervals associated with each value in quasi_ident
    :rtype: pd.Series
    """
    # Define interval boundaries from inf to sup
    values = np.arange(inf, sup + step, step)
    intervals = []

    # Assign intervals to each value in quasi_ident
    for num in quasi_ident:
        # Find the right interval
        idx = np.searchsorted(values, num, side="right") - 1
        lower = values[idx]
        upper = values[idx + 1]
        intervals.append(f"[{lower}, {upper})")

    return pd.Series(intervals, index=range(len(quasi_ident)))

@ray.remote
def compute_lipschitz_fairness(indices, predictions, features, similarity_metric, k=100):
    """
    Compute Lipschitz ratios for a chunk of data using approximate nearest neighbors.
    """
    local_max = 0.0
    knn = NearestNeighbors(n_neighbors=k, metric=similarity_metric).fit(features)

    for idx in indices:
        distances, neighbors = knn.kneighbors([features[idx]], return_distance=True)
        for distance, neighbor in zip(distances[0], neighbors[0]):
            if distance <= 1e-10:  # Skip identical features
                continue

            # Compute prediction difference
            pred_diff = entropy(predictions[idx], predictions[neighbor])
            lipschitz_ratio = pred_diff / distance
            local_max = max(local_max, lipschitz_ratio)

    return local_max

def lipschitz_fairness(predictions, features, similarity_metric='euclidean', num_workers=os.cpu_count(), k=100):
    """
    Approximate Lipschitz constant estimation using Ray and k-Nearest Neighbors.
    """
    n = len(features)
    indices = np.arange(n)
    chunk_size = math.ceil(n / num_workers)
    ray_tasks = []

    for i in range(0, n, chunk_size):
        chunk_indices = indices[i:i + chunk_size]
        ray_tasks.append(
            compute_lipschitz_fairness.remote(chunk_indices, predictions, features, similarity_metric, k)
        )

    results = ray.get(ray_tasks)
    return max(results) if results else 0.0


@ray.remote
def compute_similarity_fairness(indices, predictions, distances, neighbors):
    """
    Compute individual fairness for a chunk of data using precomputed k-NN distances and neighbors.

    Parameters:
        indices (list): Indices of the chunk.
        predictions (array-like): Model predictions for individuals.
        distances (array-like): Distances from k-NN.
        neighbors (array-like): Indices of k-NN neighbors.

    Returns:
        float: Mean individual fairness score for the chunk.
    """
    fairness_scores = []
    for idx in indices:
        # Iterate through the k-nearest neighbors (skip the first as it's the individual itself)
        for neighbor_idx, distance in zip(neighbors[idx][1:], distances[idx][1:]):
            if distance < 1e-6:  # Treat very small distances as non-zero
                distance = 1e-6
            pred_diff = abs(predictions[idx] - predictions[neighbor_idx])
            fairness_scores.append(pred_diff * distance)
    return np.mean(fairness_scores) if fairness_scores else 0.0

def similarity_fairness(predictions, features, similarity_metric='euclidean', k=100, num_workers=os.cpu_count()):
    """
    Compute individual fairness for model predictions using k-NN and Ray for parallelization.

    Parameters:
        predictions (array-like): Model predictions for individuals.
        features (array-like): Feature matrix (N x D).
        similarity_metric (str): Similarity metric ('cosine' or 'euclidean').
        k (int): Number of nearest neighbors to consider.
        num_workers (int): Number of parallel workers (default: number of CPU cores).

    Returns:
        float: Average individual fairness score (lower is better).
    """
    n = len(features)
    if k >= n:
        raise ValueError(f"Invalid k: {k}. Must be less than the number of data points: {n}.")
    
    # Fit k-NN on the features
    knn = NearestNeighbors(n_neighbors=k + 1, metric=similarity_metric).fit(features)
    distances, neighbors = knn.kneighbors(features, return_distance=True)

    # Divide indices into chunks for parallel processing
    chunk_size = max(1, n // num_workers)
    ray_tasks = [
        compute_similarity_fairness.remote(
            range(start, min(start + chunk_size, n)), predictions, distances, neighbors
        )
        for start in range(0, n, chunk_size)
    ]

    # Gather results from all chunks and compute the overall mean
    fairness_scores = ray.get(ray_tasks)
    return np.mean(fairness_scores) if fairness_scores else 0.0

@ray.remote
def compute_neighborhood_consistency_fairness(indices, predictions, distances, neighbors):
    """
    Compute Neighborhood Consistency for a chunk of data.

    Parameters:
        indices (list): Indices of the chunk.
        predictions (array-like): Model predictions for individuals.
        distances (array-like): Distances from k-NN.
        neighbors (array-like): Indices of k-NN neighbors.

    Returns:
        float: Mean neighborhood consistency score for the chunk.
    """
    consistency_scores = []
    for idx in indices:
        # Calculate consistency score for the current individual
        local_consistency = np.mean([
            abs(predictions[idx] - predictions[neighbor_idx])
            for neighbor_idx in neighbors[idx][1:]  # Skip the first neighbor (the individual itself)
        ])
        consistency_scores.append(local_consistency)

    return np.mean(consistency_scores)


def neighborhood_consistency_fairness(predictions, features, similarity_metric='euclidean', k=100, num_workers=os.cpu_count()):
    """
    Compute Neighborhood Consistency Metric using k-NN and Ray for parallelization.

    Parameters:
        predictions (array-like): Model predictions for individuals.
        features (array-like): Feature matrix (N x D).
        similarity_metric (str): Similarity metric ('cosine' or 'euclidean').
        k (int): Number of nearest neighbors to consider.
        num_workers (int): Number of parallel workers (default: number of CPU cores).

    Returns:
        float: Average neighborhood consistency score (lower is better).
    """
    # Fit k-NN on the features
    knn = NearestNeighbors(n_neighbors=k + 1, metric=similarity_metric).fit(features)
    distances, neighbors = knn.kneighbors(features, return_distance=True)

    # Divide indices into chunks for parallel processing
    n = len(features)
    chunk_size = n // num_workers
    ray_tasks = [
        compute_neighborhood_consistency_fairness.remote(
            range(start, min(start + chunk_size, n)), predictions, distances, neighbors
        )
        for start in range(0, n, chunk_size)
    ]

    # Gather results from all chunks and compute the overall mean
    consistency_scores = ray.get(ray_tasks)
    return np.mean(consistency_scores)

def calculate_total_ncp(original_df, generalized_df):
    """
    Calculate the total NCP based on the difference between the original and generalized datasets.
 
    Parameters:
    - original_df: The original dataset (DataFrame).
    - generalized_df: The generalized dataset with 'any' values (DataFrame).
 
    Returns:
    - Total NCP (float): The total Normalized Certainty Penalty.
    """
    # Step 1: Calculate distinct values for each attribute in the original dataset
    distinct_counts = original_df.nunique()
 
    # Step 2: Initialize variables for total NCP and number of tuples
    total_ncp = 0
    num_rows = original_df.shape[0]
    num_attributes = original_df.shape[1]
 
    # Step 3: Go through each tuple in the generalized dataset
    for idx in range(num_rows):
        ncp_t = 0
        for col in generalized_df.columns:
            # Step 4: If the value is 'any', compute its contribution to NCP
            if generalized_df.loc[idx, col] == 'any':
                ncp_t += 1 / distinct_counts[col]
 
        # Step 5: Divide NCP for this tuple by the total number of attributes
        ncp_t /= num_attributes
        total_ncp += ncp_t
 
    # Step 6: Return the average NCP across all tuples
    #total_ncp /= num_rows
    return total_ncp