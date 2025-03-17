# Fairness Evaluation Under Anonymization Techniques

This repository contains the implementation of the experiments and methodologies presented in the paper **"Fair Play for Individuals, Foul Play for Groups? Auditing Anonymization’s Impact on ML Fairness"**. The key contributions of this paper is a systematical investigation of the interplay between anonymization techniques and fairness in machine learning (ML) models. Through various case studies, we address several critical research questions about the effects of anonymization, record suppression, dataset characteristics, and classifier choices on fairness metrics. 

## Install Dependencies
To set up the environment and install dependencies: ```pip install -r requirements.txt```

## Running the Experiments
To run an experiment, simply use the following command in your terminal: ```python <name_of_experiment_file>.py```

Our experiments are designed to address the following key Research Questions (RQs):

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

### Summarized Results
Consolidated findings from the above experiments are presented in:  
- [2_Results_Anon_Imp_Fairness.ipynb](2_Results_Anon_Imp_Fairness.ipynb)

## Repository Structure

├──Hierachies/                              #Contains hierachies description as csv files per quasi-identifier attribute
├──Results/                                 #Contains results of each experiments as csv files 
├── 1_K_Anonymity_Impact_Fairness.py        # RQ1: 𝑘-anonymity experiments
├── 1_L_Diversity_Impact_Fairness.py        # RQ1: ℓ-diversity experiments
├── 1_T_Closeness_Impact_Fairness.py        # RQ1: 𝑡-closeness experiments
├── 1_Exp_Suppression.py                    # RQ2: Suppression experiments
├── 1_Exp_Target_Distribution.py            # RQ3: Target distribution experiments
├── 1_Exp_Data_Size_Fraction.py             # RQ4: Dataset size experiments
├── 1_Exp_Classifiers.py                    # RQ5: Generalizability experiments
├── 2_Results_Anon_Imp_Fairness.ipynb       # Consolidated results and analysis
├── config_experiments.py                   # Configuration for experiments
├── utils.py                                # Helper functions for experiments
├── adult_reconstruction.csv                # Reconsutruced adult dataset with integer target
├── compas-scores-two-years.csv             # Compas dataset
├── LICENSE                                 # License information
└── README.md                               # Project description and instructions

## Acknowledgments
This repository leverages datasets obtained via the [Folktables](https://github.com/socialfoundations/folktables) Python library. 

Anonymization methods were implemented via the [Anjana](https://github.com/IFCA-Advanced-Computing/anjana) Python library.
