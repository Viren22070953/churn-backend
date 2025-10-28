# 📊 Customer Churn Prediction System

## 📌 Overview
The **Customer Churn Prediction System** is a machine learning-based solution designed to predict whether a customer is likely to stop using a product or service.  
This project focuses on **three sectors**:
- 🎮 Online Gaming
- 📞 Telecom
- 🏦 Banking

Each sector has its own churn definition and set of features, and models are trained accordingly for **high-accuracy predictions**.

---

## 🎯 Objective
- Identify customers at risk of churning.
- Provide actionable insights for customer retention.
- Achieve **90%+ accuracy and precision**.
- Handle **sector-specific churn behavior** effectively.

---

## 🗂 Project Phases

### **Phase I – Data Collection & Preprocessing**
- **Data Sources**: Transaction logs, behavioral data, and service usage records.
- **Preprocessing Steps**:
  - Handling missing values
  - Removing duplicates
  - Encoding categorical variables
  - Normalization & scaling
  - Outlier removal
- **Tools & Libraries**: Kaggle, Pandas, NumPy, Scikit-learn

---

### **Phase II – Feature Selection & Exploration**
- **Techniques**:
  - Correlation analysis
  - Feature importance ranking
  - Visualization for trend analysis
- **EDA**: Identify patterns, trends, and anomalies.
- **Tools & Libraries**: Matplotlib, Seaborn, Scikit-learn

---

### **Phase III – Model Selection & Training**
- **Algorithms Used**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Deep Learning (Keras, TensorFlow, PyTorch)
- **Optimizations**:
  - Hyperparameter tuning
  - Cross-validation

---

### **Phase IV – Evaluation & Improvement**
- **Metrics**:
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- **Class Imbalance Handling**:
  - SMOTE oversampling
  - Weighted loss functions

---

### **Phase V – Deployment & Monitoring**
- **Backend**: Flask, FastAPI
- **Frontend**: MERN Stack (React, Node.js, Express, MongoDB)
- **Hosting**:
  - Frontend → Vercel
  - Backend → Render
  - Model → AWS
- **Maintenance**:
  - Continuous monitoring
  - Periodic updates

---

## 📂 Dataset
- **Rows**: ~38,233 (after preprocessing)
- **Features**: Reduced from **1200+** encoded features to **16 key features**
- **Target**: `churn` → (0 = Not churned, 1 = Churned)

---

## 💻 Technologies & Libraries
| Category              | Tools / Libraries |
|-----------------------|-------------------|
| **Language**          | Python            |
| **Data Processing**   | Pandas, NumPy     |
| **Visualization**     | Matplotlib, Seaborn |
| **ML Models**         | Scikit-learn, CatBoost |
| **Deep Learning**     | TensorFlow, Keras, PyTorch |
| **Balancing**         | SMOTE (imbalanced-learn) |
| **Deployment**        | Flask, FastAPI, MERN Stack |
| **Hosting**           | AWS, Render, Vercel |

---

## 🚀 How to Run
```bash
# 1️⃣ Clone the repository
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run Jupyter Notebook
jupyter notebook

# 4️⃣ Open and execute
Customer_Churn_Prediction.ipynb
