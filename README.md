# 🔄 Customer Churn Prediction

## 🚀 Live Demo
**[👉 Open Streamlit App](https://customer-churn-prediction-6rdyeqfxx72juchqmupe6x.streamlit.app/)**

## 📌 Project Overview
Machine Learning project to predict telecom customer churn 
using Python and Scikit-learn, deployed as an interactive 
Streamlit web application.

## 🛠️ Tools Used
- Python (pandas, numpy, matplotlib, seaborn)
- Scikit-learn (Logistic Regression, Random Forest)
- XGBoost
- Streamlit (deployment)
- Google Colab
- GitHub

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 79.84% | 0.8404 |
| Random Forest | 79.49% | 0.8250 |
| XGBoost | 78.64% | 0.8199 |

**Best Model: Logistic Regression**

## 🔍 Key Findings
- **26.54%** of customers are churning
- **Month-to-month** customers churn at 42.7% vs 2.8% for two-year contracts
- **Fiber Optic** customers churn most at 41.9%
- **Senior citizens** churn at 41.7% — nearly double non-seniors
- **Tenure** is the strongest negative predictor — longer stay = less churn

## 💡 Business Recommendations
- Push month-to-month customers toward annual contracts
- Investigate Fiber Optic service quality urgently
- Focus retention on first 12 months of customer journey
- Create special programs for senior citizen customers

## 📁 Project Structure
```
customer-churn-prediction/
│
├── Customer_Churn_Prediction.ipynb  # ML notebook
├── app.py                           # Streamlit app
├── churn_model.pkl                  # Saved model
├── scaler.pkl                       # Saved scaler
├── requirements.txt                 # Dependencies
└── README.md
```

## 👩‍💻 Author
Rekha Shida | Computer Engineering | Parul University
GitHub: github.com/rekhashida
LinkedIn: linkedin.com/in/rekha-sida-rs576
