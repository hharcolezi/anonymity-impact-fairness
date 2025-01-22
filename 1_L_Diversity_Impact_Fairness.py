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
from anjana.anonymity import k_anonymity, l_diversity

# ML models
from xgboost import XGBClassifier as XGB

# our generic functions
from utils import get_metrics, write_main_results_to_csv, get_generalization_levels, get_train_test_data

# our data-specific functions
from utils import clean_process_data, get_hierarchies

# our individual fairness metrics
from utils import lipschitz_fairness, similarity_fairness, neighborhood_consistency_fairness

# our config file
import config_experiments as cfg

# import folkatables
import folktables
from folktables import ACSDataSource

# ray for parallel processing (individual fairness is computationally expensive)
import ray 
import os
ray.init(num_cpus=os.cpu_count(), ignore_reinit_error=True)

def main():

	# Define the parameters
	method = 'l-diversity'
	lst_dataset = cfg.lst_dataset
	lst_sensitive_attributes = cfg.lst_sensitive_attributes
	supp_level = cfg.supp_level[1]
	lst_k = cfg.lst_k
	l_div = cfg.l_div
	max_seed = cfg.max_seed
	test_size = cfg.test_size
	fraction = cfg.fixed_fraction

	for dataset in lst_dataset:

		for protected_att in lst_sensitive_attributes[dataset]:
		
			# Main execution
			write_main_results_to_csv([], dataset=dataset, header=True)

			# read data
			if dataset == 'adult':

				# Sensitive/target 
				sens_att = "income"
		
				# Read and process the data
				data = pd.read_csv("adult_reconstruction.csv")
				threshold_target = int(data[sens_att].median())
				data = clean_process_data(data, dataset, sens_att, protected_att, threshold_target)

			elif dataset == 'ACSIncome':

				# Sensitive/target 
				sens_att = 'PINCP'

				# Read and process the data
				data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
				acs_data = data_source.get_data()
				Our_ACSIncome = folktables.BasicProblem(
														features=[
																'AGEP',
																'COW',
																'SCHL',
																'MAR',
																'OCCP',
																'POBP',
																'RELP',
																'WKHP',
																'SEX',
																'RAC1P',
																],
														target='PINCP',
														target_transform=lambda x: x,    
														group=protected_att,
														preprocess=folktables.adult_filter,
														postprocess=lambda x: np.nan_to_num(x, -1),
														)
				features, target, _ = Our_ACSIncome.df_to_pandas(acs_data)
				data = pd.concat([features, target.astype(int)], axis=1)
				threshold_target = int(data[sens_att].median())
				full_data = clean_process_data(data, dataset, sens_att, protected_att, threshold_target)

			print("Dataset: {} with Protected Attribute: {}".format(dataset, protected_att))

			# Import/defining the hierarquies for each quasi-identifier. 
			hierarchies = get_hierarchies(data, dataset)

			# Define the quasi-identifiers (all columns except the sensitive attribute)
			quasi_ident = list(set(data.columns) - {sens_att})

			# Loop over several seeds
			SEED = 0
			while SEED < max_seed:
				print(f"SEED: {SEED}")
				data = full_data.sample(frac=fraction, random_state=SEED) if dataset == 'ACSIncome' else data 

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

						if k > 1:
							# Apply l-diversity
							train_data_anon = l_diversity(train_data_anon, [], quasi_ident, sens_att, k, l_div, supp_level, hierarchies)

							# Assert that the level of l-diversity is exactly met
							actual_l_diversity = pycanon.anonymity.l_diversity(train_data_anon, quasi_ident, [sens_att])
							assert actual_l_diversity == l_div, f"l-diversity constraint not met: Expected == {l_div}, but got {actual_l_diversity}"

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

						# Separate features and target
						X_train, y_train, X_test, y_test = get_train_test_data(train_data_anon, test_data, sens_att)

						# Train the model
						model = XGB(random_state=SEED, n_jobs=-1)
						model.fit(X_train, y_train)

						# Get group fairness/utility metrics
						df_fm = test_data.copy()
						df_fm['y_pred'] = np.round(model.predict(X_test)).reshape(-1).astype(int)
						dic_metrics = get_metrics(df_fm, protected_att, sens_att)

						# Compute individual fairness metrics
						soft_ypred = model.predict_proba(X_test)
						
						asf_score = similarity_fairness(soft_ypred, X_test.values)
						dic_metrics['ASF'] = np.abs(asf_score)

						alf_score = lipschitz_fairness(soft_ypred, X_test.values)
						dic_metrics['ALF'] = np.abs(alf_score)

						ncf_score = neighborhood_consistency_fairness(soft_ypred, X_test.values)
						dic_metrics['NCF'] = np.abs(ncf_score)
						print(dic_metrics)

						# Write results to csv
						write_main_results_to_csv([SEED, protected_att, sens_att, method, k, l_div] + list(dic_metrics.values()), dataset=dataset)

					except Exception as e:
							print(f"An error occurred for SEED {SEED}, k {k}: {e}")
							continue
					
				SEED += 1
				print('-------------------------------------------------------------\n')
			print('=============================================================\n')
		print('################################################################\n')
	ray.shutdown()
	
if __name__ == "__main__":
    main()