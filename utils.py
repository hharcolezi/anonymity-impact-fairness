
import typing
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import csv

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

def write_suppression_results_to_csv(values, header=False):
    """Write the results to a csv file."""

    file_path = "results/anonymity_impact_fairness_suppression.csv"
    # Check if the file exists and is empty
    file_exists = os.path.isfile(file_path)
    file_empty = os.stat(file_path).st_size == 0 if file_exists else True

    with open(file_path, mode='a', newline='') as scores_file:
        scores_writer = csv.writer(scores_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header and file_empty:# Write header if specified and file is empty
            scores_writer.writerow(["SEED", "dataset", "protected_att", "target", "method", "anon_parameter", "supp_level", "SPD", "EOD", "MAD", "PED", "PRD", "ACC", "f1", "Precision", "Recall", "ROC_AUC", "CM"])
        if not header: # Write the actual values
            scores_writer.writerow(values)

def write_results_to_csv(values, header=False):
    """Write the results to a csv file."""

    file_path = "results/anonymity_impact_fairness.csv"
    # Check if the file exists and is empty
    file_exists = os.path.isfile(file_path)
    file_empty = os.stat(file_path).st_size == 0 if file_exists else True

    with open(file_path, mode='a', newline='') as scores_file:
        scores_writer = csv.writer(scores_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header and file_empty:# Write header if specified and file is empty
            scores_writer.writerow(["SEED", "dataset", "protected_att", "target", "method", "k_parameter", "anon_parameter", "SPD", "EOD", "MAD", "PED", "PRD", "ACC", "f1", "Precision", "Recall", "ROC_AUC", "CM"])
        if not header: # Write the actual values
            scores_writer.writerow(values)

def clean_process_adult_data(data, sens_att, protected_att='gender', threshold_target=50000):
    """Clean and preprocess the adult dataset."""
    
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
    return data

def get_hierarchies_adult(data):
    """Generate/read hierarchies for quasi-identifiers based on provided data."""
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