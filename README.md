# 🏦 **PRODIGY_DS_03: Bank Marketing - Decision Tree Classifier**

## 🎯 **Project Overview**
Comprehensive machine learning pipeline to predict customer purchase behavior using Decision Tree Classifier on the UCI Bank Marketing dataset for Portuguese bank term deposit campaigns.

## 📊 **Dataset Information**
- **Source**: Prodigy InfoTech / [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Records**: 45,211 bank clients
- **Original Features**: 16 demographic/behavioral + 1 target (`y`)
- **Key Features**: `age`, `job`, `marital`, `education`, `duration`, `campaign`, `pdays`, `poutcome`
- **Target**: `y` (no/yes term deposit subscription) - **11.7% positive rate**

## 🛠️ **Tools & Technologies**
   Python 3.9+, pandas, NumPy
scikit-learn (DecisionTreeClassifier)
matplotlib, seaborn, Jupyter Notebook
ucimlrepo, joblib


## 📋 **Methodology**

### 1. **Data Loading & Initial Exploration**
- Loaded UCI dataset (45,211 × 17 features)
- Identified severe class imbalance (88.3% No, 11.7% Yes)
- Analyzed 10 categorical + 7 numerical features

### 2. **Data Preprocessing**
- **Categorical Encoding**: LabelEncoder on `job`, `marital`, `education`, `contact`, `month`, `day`, `poutcome`
- **Target Encoding**: `y` → {no:0, yes:1}
- **Outlier Removal**: Duration IQR method
- **Result**: Clean dataset ready for modeling

### 3. **Model Training**
```
DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
```


###4. **Model Evaluation**
      Test Accuracy: 89.2%
      ROC-AUC: 0.91
      Precision(Yes): 0.58, Recall(Yes): 0.52

###5. **Visualization Suite (8 Charts)**
     Target distribution, age/job distributions
     Duration/campaign boxplots, feature importance
     Confusion matrix, correlation heatmap
      Decision tree structure

🔍 Key Findings
📈 Overall Statistics
Total Clients: 45,211
Purchased: 5,285 (11.7%)
Not Purchased: 39,926 (88.3%)
Test Accuracy: 89.2%

🎯 Critical Purchase Factors (Ranked by Feature Importance)

| Rank | Feature  | Importance | Business Insight                          |
| ---- | -------- | ---------- | ----------------------------------------- |
| 1    | pdays    | 0.42       | Days since last contact - recency matters |
| 2    | duration | 0.28       | Longer calls → higher conversion          |
| 3    | age      | 0.09       | 30-50 age group most responsive           |
| 4    | poutcome | 0.07       | Previous campaign success predictor       |
| 5    | campaign | 0.05       | 1-3 contacts optimal                      |


🚀 How to Replicate
Prerequisites
bash
```
pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 matplotlib==3.7.2 seaborn==0.12.2 ucimlrepo==0.0.4 jupyter==1.0.0 joblib==1.3.0
Quick Start
bash
# 1. Clone repository
gitclone https://github.com/YOUR_USERNAME/PRODIGY_DS_03.git
cd PRODIGY_DS_03

##2. Launch Jupyter
jupyter notebook

#$ 3. Open notebook & run all cells
Bank_Marketing_Decision_Tree.ipynb
```
