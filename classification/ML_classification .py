#!/usr/bin/env python
# coding: utf-8

# # client_attrition for classification
# 
# The task is to apply various ML algorithms to build a model explaining whether a particular person closed the credit card account based on the training sample and generate predictions for all observations from the test sample.
# 
# The dataset includes 10127 observations in the training sample and 5063 in the test sample and the following columns:
# 
# - `customer_id`: unique observation identifier
# - `customer_age`: age of the customer in years
# - `customer_sex`: gender of the customer
# - `customer_number_of_dependents`: number of dependents on the customer
# - `customer_education`: education level of the customer
# - `customer_civil_status`: civil status of the customer
# - `customer_salary_range`: range of the annual salary of the customer
# - `customer_relationship_length`: length of the customer’s relationship with the bank in months
# - `customer_available_credit_limit`: available limit on the customer’s credit card account
# - `credit_card_classification`: classification of the card (Blue, Silver, Gold, Platinum)
# - `total_products`: total number of products held by the customer in the bank
# - `period_inactive`: period in the last year when the customer was inactive (in months)
# - `contacts_in_last_year`: number of contacts with the customer in the last year
# - `credit_card_debt_balance`: total card debt balance on the credit card account
# - `remaining_credit_limit`: remaining limit on the customer’s credit card account (average in the last year)
# - `transaction_amount_ratio`: ratio of the total amount of transactions in the 4th quarter against the 1st quarter
# - `total_transaction_amount`: total amount of transactions in the last year
# - `total_transaction_count`: total number of transactions in the last year
# - `transaction_count_ratio`: ratio of the total count of transactions in the 4th quarter against the 1st quarter
# - `average_utilization`: average card utilization (percentage used of the total limit)
# - `account_status`: customer account status: closed, open (outcome variable, only in the training sample)
# 
# 
# ## Select the best algorithm
# 

# ## Libraries

# In[1]:


import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# ## Data

# In[2]:


# Load data
train_data = pd.read_csv('client_attrition_train.csv')
test_data = pd.read_csv('client_attrition_test.csv')


# In[3]:


# Display the first few rows
train_data.head()


# In[4]:


# Display data types of columns in train_data
print(train_data.dtypes)


# In[5]:


# Display the first few rows
test_data.head()


# ##  Data Cleaning

# In[6]:


# Check for missing values in train data
print("Missing values in the training data:")
print(train_data.isnull().sum())

# Fill missing numeric values with mean
train_data['customer_age'].fillna(train_data['customer_age'].mean(), inplace=True)
train_data['customer_available_credit_limit'].fillna(train_data['customer_available_credit_limit'].mean(), inplace=True)
train_data['remaining_credit_limit'].fillna(train_data['remaining_credit_limit'].mean(), inplace=True)
train_data['transaction_amount_ratio'].fillna(train_data['transaction_amount_ratio'].mean(), inplace=True)
train_data['total_transaction_amount'].fillna(train_data['total_transaction_amount'].mean(), inplace=True)
train_data['transaction_count_ratio'].fillna(train_data['transaction_count_ratio'].mean(), inplace=True)
train_data['average_utilization'].fillna(train_data['average_utilization'].mean(), inplace=True)

# Fill missing categorical values with most frequent value
train_data['customer_sex'].fillna(train_data['customer_sex'].mode()[0], inplace=True)
train_data['customer_salary_range'].fillna(train_data['customer_salary_range'].mode()[0], inplace=True)


# In[7]:


# Ensure there are no missing values in train data
print(train_data.isnull().sum())


# ## Data preprocessing

# In[8]:


# Separates the input features (x) from the target variable (y)
x = train_data.drop('account_status', axis=1)
y = train_data['account_status']


# ### Label Encoder

# In[9]:


# Encode 
label_encoder = LabelEncoder()
x[['customer_sex','customer_education','customer_civil_status','customer_salary_range','credit_card_classification']] = x[['customer_sex','customer_education','customer_civil_status','customer_salary_range','credit_card_classification']].apply(label_encoder.fit_transform)


# ### Balance

# In[10]:


# Check balance
class_id_distribution = y.value_counts()
print(class_id_distribution)


# In[11]:


# Balance the data
smote = SMOTE(random_state=42)
x_balanced, y_balanced = smote.fit_resample(x, y)


# ### Feature selection

# In[12]:


# Select the top k features
k = 20  
selector = SelectKBest(score_func=f_classif, k=k)
x_selected = selector.fit_transform(x_balanced, y_balanced)
selected_indices = selector.get_support(indices=True)
selected_features = x.columns[selected_indices]
print("Selected Features:")
print(selected_features)


# ### Scale the data

# In[13]:


# Scaling
scaler = StandardScaler()
x_selected_scaled = scaler.fit_transform(x_selected)


# ### Split

# In[14]:


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_selected_scaled, y_balanced, test_size=0.2, random_state=42)


# ## Final model 

# In[15]:


# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_balanced)

# Fit the CatBoost model on the training data
catboost = CatBoostClassifier()
catboost.fit(x_selected_scaled, y_encoded)

# CatBoost Classifier
catboost.fit(x_train, y_train)
catboost_predictions = catboost.predict(x_test)
catboost_score = balanced_accuracy_score(y_test, catboost_predictions)
print(f"CatBoost Classifier: {catboost_score:.4f} ({catboost_score*100:.2f}%)")

# Filter the selected features that exist in both training and test data
common_features = selected_features.intersection(x.columns)

# Define the categorical features
categorical_features = ['customer_sex', 'customer_education', 'customer_civil_status', 'customer_salary_range', 'credit_card_classification']

# Apply one-hot encoding to the test data using only the common features
x_test_encoded = pd.get_dummies(test_data[common_features], columns=categorical_features)

# Reorder the test data columns to match the training data
x_test_encoded = x_test_encoded.reindex(columns=x.columns, fill_value=0)

# Scale the test data using the trained scaler
x_test_scaled = scaler.transform(x_test_encoded)

# Generate predictions using the trained CatBoost model
predictions = catboost.predict(x_test_scaled)

# Print the predictions
print(predictions)


# ## CSV file

# In[16]:


# Create a DataFrame with the predictions
predictions_df = pd.DataFrame(predictions, columns=['predictions'])

# Save the predictions to a CSV file
predictions_df.to_csv('predictions_classification.csv', index=False)

