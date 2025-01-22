# Fairness Evaluation Under Anonymization Techniques

This repository investigates the interplay between anonymization techniques and fairness in machine learning (ML) models. Through various case studies presented in Python notebooks, we address several critical research questions about the effects of anonymization, suppression, dataset characteristics, and classifier choices on fairness metrics. 
The results and findiings of this research are provided in the paper : ***Title and link later*** 

## Research Questions (RQs)

### RQ1: Impact of Anonymization Techniques on Fairness
How do different anonymization techniques (𝑘-anonymity, ℓ-diversity, and 𝑡-closeness) and their parameters affect the fairness of ML models?  
- Experiments:  
  - [1_K_Anonymity_Impact_Fairness.ipynb](1_K_Anonymity_Impact_Fairness.ipynb)  
  - [1_L_Diversity_Impact_Fairness.ipynb](1_L_Diversity_Impact_Fairness.ipynb)  
  - [1_T_Closeness_Impact_Fairness.ipynb](1_T_Closeness_Impact_Fairness.ipynb)  

### RQ2: Effect of Suppression on Fairness
How does varying the record-level suppression threshold during anonymization impact fairness, particularly for sub-populations?  
- Experiment:  
  - [1_Exp_Suppression.ipynb](1_Exp_Suppression.ipynb)  

### RQ3: Influence of Target Distribution
What is the impact of altering the target distribution on fairness metrics, specifically by varying the threshold for binarizing the income variable?  
- Experiment:  
  - [1_Exp_Target_Distribution.ipynb](1_Exp_Target_Distribution.ipynb)  

### RQ4: Role of Dataset Size
How does dataset size mediate the trade-offs between privacy, fairness, and utility?  
- Experiment:  
  - [1_Exp_Data_Size_Fraction.ipynb](1_Exp_Data_Size_Fraction.ipynb)  

### RQ5: Generalizability Across Classifiers
Do fairness results observed using XGBoost generalize across other ML classifiers, such as Random Forests and Neural Networks?  
- Experiment:  
  - [1_Exp_Classifiers.ipynb](1_Exp_Classifiers.ipynb)  

### Summarized Results
Consolidated findings from the above experiments are presented in:  
- [2_Results_Anon_Imp_Fairness.ipynb](2_Results_Anon_Imp_Fairness.ipynb)

## Repository Structure

├──Hierachies/                              #Contains hierachies description as csv files per quasi-identifier attribute

├──Results/                                 #Contains results of each experiments as csv files 

├── 1_K_Anonymity_Impact_Fairness.ipynb     # RQ1: 𝑘-anonymity experiments

├── 1_L_Diversity_Impact_Fairness.ipynb     # RQ1: ℓ-diversity experiments

├── 1_T_Closeness_Impact_Fairness.ipynb     # RQ1: 𝑡-closeness experiments

├── 1_Exp_Suppression.ipynb                 # RQ2: Suppression experiments

├── 1_Exp_Target_Distribution.ipynb         # RQ3: Target distribution experiments

├── 1_Exp_Data_Size_Fraction.ipynb          # RQ4: Dataset size experiments

├── 1_Exp_Classifiers.ipynb                 # RQ5: Generalizability experiments

├── 2_Results_Anon_Imp_Fairness.ipynb       # Consolidated results and analysis

├── config_experiments.py                   # Configuration for experiments

├── utils.py                                # Helper functions for experiments

├── adult_reconstruction.csv                # Dataset (Reconsutruced adult dataset with integer target) for anonymization and fairness studies

├── LICENSE                                 # License information

└── README.md                               # Project description and instructions

# Clone the repository
git clone https://github.com/anonymity-impact-fairness.git

cd anonymity-impact-fairness

# Install dependencies (list dependencies in README or use requirements.txt)
pip install -r requirements.txt

# Run specific experiments
# Replace <notebook_name> with the name of the desired notebook (e.g., 1_K_Anonymity_Impact_Fairness.ipynb)
jupyter notebook <notebook_name>.ipynb

# Example: Run the experiment for k-Anonymity's impact on fairness
jupyter notebook 1_K_Anonymity_Impact_Fairness.ipynb

# Analyze consolidated results
# Open the notebook for results and analysis
jupyter notebook 2_Results_Anon_Imp_Fairness.ipynb

