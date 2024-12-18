# Churn_Risk_Classifier

## Overview
This project focuses on predicting customer churn using machine learning techniques. Understanding customer churn is essential for businesses aiming to improve retention rates and enhance customer satisfaction. 

## Dataset 
The dataset used for this project includes various features that influence customer retention, such as:

- **CustomerID**: Unique identifier for each customer
- **Gender**: Gender of the customer
- **SeniorCitizen**: Indicates if the customer is a senior citizen (1 = Yes, 0 = No)
- **Partner**: Whether the customer has a partner (Yes/No)
- **Dependents**: Whether the customer has dependents (Yes/No)
- **Tenure**: Duration of the customer's relationship with the company (in months)
- **PhoneService**: Whether the customer has phone service (Yes/No)
- **MultipleLines**: Whether the customer has multiple lines (Yes/No)
- **InternetService**: Type of internet service (DSL/Fiber optic/No)
- **OnlineSecurity**: Whether the customer has online security (Yes/No)
- **OnlineBackup**: Whether the customer has online backup (Yes/No)
- **DeviceProtection**: Whether the customer has device protection (Yes/No)
- **TechSupport**: Whether the customer has tech support (Yes/No)
- **StreamingTV**: Whether the customer has streaming TV (Yes/No)
- **StreamingMovies**: Whether the customer has streaming movies (Yes/No)
- **Contract**: Type of contract (Month-to-month/One year/Two year)
- **PaperlessBilling**: Whether the customer has paperless billing (Yes/No)
- **PaymentMethod**: Payment method used by the customer
- **MonthlyCharges**: Monthly charges of the customer
- **TotalCharges**: Total charges incurred by the customer
- **Churn**: Whether the customer churned (Yes/No)

# Data Preprocessing
The following preprocessing steps were undertaken to prepare the data for modelling

1. ** Data Cleaning**:Handled missing values and ensudred data integrity.
2. **Feature Engineering**:Created new features from existing data to better capture customer behavior.
3. **Encoding Categorical variables **:Applies one-hot encoding to transform categorical variables into numerical format for model training.
4. **Data Scaling**:Standardized numerical features to bring them onto a similar scale,which is critical for certain algorithms.

## Models Implemented 
The project evaluated the following machine learning models:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **AdaBoost Classifier**
- **Gradient Boosting Classifier**
- **XGBoost Classifier**

## Model Performance 
The models were evaluated based on accuracy, with the following results:

1. **Logistic Regression**: **Accuracy**: 0.80 
2. **Decision Tree Classifier**: **Accuracy**: 0.81 
3. **Random Forest Classifier**: **Accuracy**: 0.82 
4. **AdaBoost Classifier**: **Accuracy**: 0.82 
5. **Gradient Boosting Classifier**: **Accuracy**: 0.82 
6. **XGBoost Classifier**: **Accuracy**: 0.84 🎉
7. **Final Gradient Boosting Classifier Model**: **Accuracy**: 0.83

## Conclusion 
The **XGBoost Classifier** outperformed other models with an impressive accuracy of **84%**, effectively identifying customers at risk of churning. This project not only enhanced my understanding of customer behavior but also provided valuable insights into various machine learning methodologies and their applications in business contexts. 
  
