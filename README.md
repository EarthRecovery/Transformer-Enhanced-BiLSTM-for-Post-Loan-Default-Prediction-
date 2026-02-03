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
- **Source**:
```text
https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
```
- **Label Definition**:
  - Default (`y=1`): `Current Loan Delinquency Status ≥ 3`
  - Non-default (`y=0`): otherwise
- **Balancing**: Random undersampling in the training set to achieve a 1:1 ratio
- **Key Features**:
  - Interest Bearing UPB-Delta
  - Current Actual UPB-Delta
  - Estimated Loan to Value (ELTV)
  - Additional loan status, modification flags, interest rates, etc. (total 238 features)
- **File Naming Constraint**: Only dataset files with names starting with `historical_data_time` are supported. Please copy the txt file into Data folder(not the data folder in the Project folder)

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

## Parameters Reference
All parameters are defined in `Project/Paramaters/Parameter.json`.

1. `ExpName`: Experiment name used in output folder naming.  
   Example: `"NewMethodtest"`.
1. `HistoryHeader`: Full header list for the history dataset (used when loading raw data).  
   Format: comma-separated string of column names.
1. `OriginationHeader`: Full header list for the origination dataset.  
   Format: comma-separated string of column names.
1. `HistorySelectedHeader`: Subset of history columns used for modeling.  
   Format: comma-separated string of column names.
1. `JumpNumber`: Time step jump size when constructing sequences.  
   Type: integer.
1. `PredictMonth`: Number of future months used as label window.  
   Type: integer.
1. `Split`: Train/test split option (currently used as a flag in the pipeline).  
   Type: integer (typically `1`).
1. `Verbose`: Verbosity level for training logs.  
   Type: integer (`0` or `1`).
1. `RandomSeed`: Random seed for reproducibility.  
   Type: integer.
1. `WindowSize`: Sequence window length (total months per sample).  
   Type: integer.
1. `DataSet`: Input dataset file name under the `Data/` directory.  
   Example: `"historical_data_time_2017Q1.txt"`.
1. `Nrows`: Number of rows to load from the dataset (for sampling/debug).  
   Type: integer.
1. `Normalize`: Whether to apply MinMax scaling to numeric features.  
   Type: `0` (off) or `1` (on).
1. `InsideDataFilename`: Fixed output folder name under `Project/data/`.  
   Use `""` to auto-name with `ExpName-Time`.
1. `Epochs`: Training epochs.  
   Type: integer.
1. `BatchSize`: Training batch size.  
   Type: integer.
1. `EvaluationModelNames`: Models to train/evaluate.  
   Allowed values (from `Project/Models`):  
   `["BERTModel","BiLSTMModel","BiLSTMMTEModel","CNNModel","GRUModel","LSTMModel","RNNModel","TBiLSTMModel"]`.
1. `EvaluationTimes`: How many repeated evaluation runs to average.  
   Type: integer.
1. `EnableSHAP`: Enable/disable SHAP explainability.  
   Type: `true` or `false`.
1. `SHAPModelNames`: Subset of models to run SHAP on.  
   Use `[]` to run SHAP for all evaluated models.

---

## Environment Setup (Conda)
1. Create and activate a conda environment:
```bash
conda create -n resebilstm python=3.10 -y
conda activate resebilstm
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. start project
```bash
python Project/Main/main.py
```


@article{yang2025resebilstm,
  title={Transforming Credit Risk Analysis: A Time-Series-Driven ResE-BiLSTM Framework for Post-Loan Default Detection},
  author={Yang, Yue and Lin, Yuxiang and Zhang, Ying and Su, Zihan and Goh, Chang Chuan and Fang, Tangtangfang and Bellotti, Anthony Graham and Lee, Boon Giin},
  journal={arXiv preprint arXiv:2508.00415},
  year={2025}
}