# Fairness and Anonymization in Machine Learning: Research Questions and Experiments

This repository contains the code and experiments for addressing key research questions related to the intersection of **fairness**, **anonymization**, and **machine learning (ML)**. The experiments are implemented in Python through a series of Jupyter notebooks.

## Research Questions

This study aims to answer the following research questions (RQs):

### **RQ1: How do different anonymization techniques and anonymity levels affect the fairness of ML models?**
This research question evaluates the impact of three widely used anonymization techniques:
- **k-anonymity**
- **ℓ-diversity**
- **t-closeness**

By systematically varying their respective privacy parameters (\(k\), \(l\), and \(t\)), we investigate:
- How these techniques influence fairness metrics.
- Whether specific configurations disproportionately impact certain demographic groups.

Experiments for RQ1 are presented in **Section 4.1** of the paper.

---

### **RQ2: What is the impact of varying record-level suppression in anonymization on the fairness of ML models?**
Suppression often targets outlier data, which may disproportionately affect certain sub-populations. This research question explores:
- The effect of varying suppression thresholds (removing rows) on fairness metrics.
- The trade-offs between robust privacy protection and equitable treatment across demographic groups.

Experiments for RQ2 are detailed in **Section 4.2** of the paper.

---

### **RQ3: What is the impact of varying target distributions on the fairness of ML models?**
Changes in the target distribution, such as adjusting thresholds for binarizing the income variable, can influence:
- The balance between positive and negative outcomes in the dataset.
- Fairness metrics across demographic groups.

Experiments for RQ3 are presented in **Section 4.3** of the paper.

---

### **RQ4: How does dataset size influence the relationship between anonymization and fairness in ML models?**
This research question investigates how dataset size mediates the trade-offs between:
- Privacy
- Fairness
- Utility

By systematically varying the data fraction, we analyze how sample size impacts these trade-offs.

Experiments for RQ4 are detailed in **Section 4.4** of the paper.

---

### **RQ5: To what extent are the fairness results obtained with XGBoost representative across different ML classifiers?**
The default experiments in RQ1–RQ4 use **XGBoost**. This question examines:
- Whether the fairness results generalize to other ML classifiers (e.g., Random Forest, Neural Networks).
- The consistency of trends observed with XGBoost across classifiers.

Experiments for RQ5 are presented in **Section 4.5** of the paper.

---

## Repository Structure

The repository is organized as follows:

