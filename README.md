# Transforming Credit Risk Analysis: ResE-BiLSTM for Post-Loan Default Detection

## 📌 Project Overview
This repository implements the **Residual-enhanced Encoder Bidirectional LSTM (ResE-BiLSTM)** framework proposed in the paper:  
**"Transforming Credit Risk Analysis: A Time-Series-Driven ResE-BiLSTM Framework for Post-Loan Default Detection"**.  

The framework addresses **post-loan default prediction** using **monthly repayment time-series data**.  
It combines a **multi-head attention–based residual encoder** for global feature extraction and a **BiLSTM** for capturing bidirectional temporal dependencies, with residual connections to improve stability and convergence.  
The model significantly outperforms common baselines including LSTM, BiLSTM, GRU, CNN, and RNN.

---

## ✨ Key Features
- **Time-series driven**: Sliding window processing of monthly repayment data with a 14-month input, 2-month blank period, and 3-month observation window.
- **Multi-head Attention + Residual Connections**: Captures global dependencies and mitigates gradient vanishing.
- **Bidirectional LSTM**: Models both past and future temporal dependencies.
- **SHAP Interpretability**: Explains feature importance for regulatory compliance and trust building.
- **Multi-metric evaluation**: Accuracy, Precision, Recall, F1, AUC, and Average Ranking (AvgR).
- **Ablation studies**: Evaluates the contribution of each architectural component.

---

## 📂 Dataset
**Freddie Mac Single-Family Loan-Level Dataset**
- **Period**: 2009Q1 – 2019Q4  
- **Cohorts**: 44 independent quarterly cohorts (1,000,000 monthly loan records per cohort)  
- **Label Definition**:
  - Default (`y=1`): `Current Loan Delinquency Status ≥ 3`
  - Non-default (`y=0`): otherwise
- **Balancing**: Random undersampling in the training set to achieve a 1:1 ratio
- **Key Features**:
  - Interest Bearing UPB-Delta
  - Current Actual UPB-Delta
  - Estimated Loan to Value (ELTV)
  - Additional loan status, modification flags, interest rates, etc. (total 238 features)

---


## 🔍 Experimental Highlights
- **Highest Accuracy** in 38/44 cohorts (86.36%)
- **Highest Recall** in 37/44 cohorts — significantly reducing false negatives
- **Top AvgR** ranking in 90%+ of yearly cohort groupings
- **Ablation results**:
  - Removing the Feedforward Network causes the largest performance drop
  - Residual connections improve recall and minority-class recognition

---

## 📊 Interpretability
Using **SHAP** feature attribution:
- **Top Features**:
  - Interest Bearing UPB-Delta
  - Current Actual UPB-Delta
  - ELTV
- Feature importance varies over time; ResE-BiLSTM utilizes both recent and distant historical information.
- Certain features (e.g., disaster-related delinquency) have weaker contributions.

---

@article{yang2025resebilstm,
  title={Transforming Credit Risk Analysis: A Time-Series-Driven ResE-BiLSTM Framework for Post-Loan Default Detection},
  author={Yang, Yue and Lin, Yuxiang and Zhang, Ying and Su, Zihan and Goh, Chang Chuan and Fang, Tangtangfang and Bellotti, Anthony Graham and Lee, Boon Giin},
  journal={arXiv preprint arXiv:2508.00415},
  year={2025}
}
