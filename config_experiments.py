# config_experiments.py

lst_dataset = ['adult', 'ACSIncome', 'compas']
lst_sensitive_attributes = {
                            'adult': ['gender', 'race'],
                            'ACSIncome': ['SEX', 'RAC1P'],
                            'compas': ['sex', 'race'],
                        }
supp_level = [10, 20, 30, 40, 50]
lst_fraction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
lst_deciles = range(10, 100, 10)
lst_k = list(range(1, 11)) + [25, 50, 75, 100]
l_div = 2
lst_t = [0.45, 0.50, 0.55]
max_seed = 40
test_size = 0.2
fixed_k = 10
fixed_t = 0.5
fixed_l = 2
fixed_fraction = 0.1
