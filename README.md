# Fairness Evaluation Under Anonymization Techniques

This repository contains the implementation of the experiments and methodologies presented in the paper **"Fair Play for Individuals, Foul Play for Groups? Auditing Anonymization’s Impact on ML Fairness"**. The key contributions of this paper is a systematical investigation of the interplay between anonymization techniques and fairness in machine learning (ML) models. Through various case studies, we address several critical research questions about the effects of anonymization, record suppression, dataset characteristics, and classifier choices on fairness metrics. 

## Install Dependencies
To set up the environment and install dependencies: ```pip install -r requirements.txt```

## Running the Experiments
Each script answers a specific research question (RQ). Run any experiment with: ```python <experiment_file>.py```

### RQ1: Impact of Anonymization Techniques on Fairness
How do different anonymization techniques (𝑘-anonymity, ℓ-diversity, and 𝑡-closeness) and their parameters affect the fairness of ML models?  
- Experiments:  
  - [1_K_Anonymity_Impact_Fairness.py](1_K_Anonymity_Impact_Fairness.py)  
  - [1_L_Diversity_Impact_Fairness.py](1_L_Diversity_Impact_Fairness.py)  
  - [1_T_Closeness_Impact_Fairness.py](1_T_Closeness_Impact_Fairness.py)  

### RQ2: Effect of Suppression on Fairness
How does varying the record-level suppression threshold during anonymization impact fairness, particularly for sub-populations?  
- Experiment:  
  - [1_Exp_Suppression.py](1_Exp_Suppression.py)  

### RQ3: Influence of Target Distribution
What is the impact of altering the target distribution on fairness metrics, specifically by varying the threshold for binarizing the income variable?  
- Experiment:  
  - [1_Exp_Target_Distribution.py](1_Exp_Target_Distribution.py)  

### RQ4: Role of Dataset Size
How does dataset size mediate the trade-offs between privacy, fairness, and utility?  
- Experiment:  
  - [1_Exp_Data_Size_Fraction.py](1_Exp_Data_Size_Fraction.py)  

### RQ5: Generalizability Across Classifiers
Do fairness results observed using XGBoost generalize across other ML classifiers, such as Random Forests and Neural Networks?  
- Experiment:  
  - [1_Exp_Classifiers.py](1_Exp_Classifiers.py)  

### Result Aggregation
Consolidated findings from the above experiments are presented in:  
- [2_Results_Anon_Imp_Fairness.ipynb](2_Results_Anon_Imp_Fairness.ipynb)

## Repository Structure

├── data/                         	    # Datasets used in the experiments
│   ├── adult_reconstruction.csv  	    # Reconstructed ACSIncome dataset
│   └── compas-scores-two-years.csv     # COMPAS dataset

├── hierarchies/                 	    # CSV hierarchies for QI generalization
│   ├── ACSIncome  	    				
│   ├── adult					  	    
│   └── compas						    

├── results/                     	    # Output CSVs from all experiments

├── 1_K_Anonymity_Impact_Fairness.py    # RQ1: 𝑘-anonymity experiment
├── 1_L_Diversity_Impact_Fairness.py    # RQ1: ℓ-diversity experiment
├── 1_T_Closeness_Impact_Fairness.py    # RQ1: 𝑡-closeness experiment

├── 1_Exp_Suppression.py                # RQ2: Suppression threshold impact
├── 1_Exp_Target_Distribution.py        # RQ3: Effect of target binarization
├── 1_Exp_Data_Size_Fraction.py         # RQ4: Dataset size variation
├── 1_Exp_Classifiers.py                # RQ5: Model generalization study

├── 2_Results_Anon_Imp_Fairness.ipynb   # Notebook with aggregated results
├── config_experiments.py               # Configuration file for all experiments
├── utils.py                            # Utility functions
├── hierarchy_gen_ACSIncome.py          # Generates hierarchies for ACSIncome

├── requirements.txt                    # Required Python libraries
├── LICENSE                             # License for the repository
└── README.md                           # Project overview and usage guide


## Acknowledgments
The ```ACSIncome``` dataset was obtained via the [Folktables](https://github.com/socialfoundations/folktables) Python library. 

Anonymization methods were implemented via the [Anjana](https://github.com/IFCA-Advanced-Computing/anjana) Python library.
