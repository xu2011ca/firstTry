import numpy as np
import pandas as pd

#import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, recall_score, precision_score, accuracy_score

from boruta import BorutaPy
import shap

import matplotlib.pyplot as plt

import joblib

def make_target(row):
   if (row['Bullied_on_school_property_in_past_12_months'] == 'Yes'
       or row['Bullied_not_on_school_property_in_past_12_months'] == 'Yes' \
       or row['Cyber_bullied_in_past_12_months'] == 'Yes'):
      return 1
   return 0

#clean data
df = pd.read_csv('Bullying_2018.csv', delimiter=';')
df.drop('record', axis=1, inplace=True)

# To simplify our analysis, we may just drop the missing rows
# However, there are columns with empty strings.
#So, we may first replace this values with nan
df.replace({' ':np.nan}, inplace=True)
df.dropna(inplace=True)

# Creating our target variable
df['target'] = df.apply(lambda row: make_target(row), axis=1)
print(df.target.value_counts(normalize=True))

# Dropping the bullying features
df.drop(['Bullied_on_school_property_in_past_12_months',
         'Bullied_not_on_school_property_in_past_12_months',
         'Cyber_bullied_in_past_12_months'],
        axis=1,
        inplace=True)

# Checking if there are missing values
print(df.isna().sum())

#Model Development

# Setting the columns to ordinal or categorical
# Here, categorical columns with a sense of order (age, number of times that something hapenned) were set to ordinal

categorical_columns = [
    'Sex',
    'Felt_lonely',
    'Other_students_kind_and_helpful',
    'Parents_understand_problems',
    'Most_of_the_time_or_always_felt_lonely',
    'Missed_classes_or_school_without_permission',
    'Were_underweight',
    'Were_overweight',
    'Were_obese'
]

ordinal_columns = [
    'Custom_Age',
    'Physically_attacked',
    'Physical_fighting',
    'Close_friends',
    'Miss_school_no_permission'
]

# Creating the mapping order to ordinal columns

ordinal_cols_mapping = [
    ['11 years old or younger', '12 years old', '13 years old', '14 years old', '15 years old', '16 years old', '17 years old', '18 years old or older'],
    ['0 times', '1 time', '2 or 3 times', '4 or 5 times', '6 or 7 times', '8 or 9 times', '10 or 11 times', '12 or more times'],
    ['0 times', '1 time', '2 or 3 times', '4 or 5 times', '6 or 7 times', '8 or 9 times', '10 or 11 times', '12 or more times'],
    ['0', '1', '2', '3 or more'],
    ['0 days', '1 or 2 days', '3 to 5 days', '6 to 9 days', '10 or more days']
]

# Constructing the preprocessing pipeline

categorical_transformer = Pipeline(
    steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ]
)

ordinal_transformer = Pipeline(
    steps=[
        ('encoder', OrdinalEncoder(categories=ordinal_cols_mapping, handle_unknown='use_encoded_value', unknown_value=-1))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_columns),
        ('ord', ordinal_transformer, ordinal_columns)
    ]
)

# Defining the model pipeline

model_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight='balanced',
    n_jobs=-1,
    verbose=True
)

model_pipeline_rf = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('rf', model_rf)
    ]
)

# With this set_output API we are able to track the feature names
#which the pipeline outputs
model_pipeline_rf.set_output(transform='pandas')

# Train/test split

X = df.drop('target', axis=1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.3,
    random_state=123
)
# Fitting the model

model_pipeline_rf.fit(X_train, y_train)

#Model Evaluation

y_pred = model_pipeline_rf.predict(X_test)

print('Accuracy score:', recall_score(y_test, y_pred))
print('Recall score:', recall_score(y_test, y_pred, average='macro'))
print('Precision score:', precision_score(y_test, y_pred, average='macro'))
print('AUC score:', roc_auc_score(y_test, y_pred))
'''
disp = ConfusionMatrixDisplay.from_estimator(estimator=model_pipeline_rf,
                                             X=X_test,
                                             y=y_test,
                                             normalize='true')


#Feature Importance
features = model_pipeline_rf['rf'].feature_names_in_
importances = model_pipeline_rf['rf'].feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12,10))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
'''

#Boruta

# Creating DataFrame for categorical test and train
X_train_cat = X_train[categorical_columns]
X_test_cat = X_test[categorical_columns]

# Creating DataFrame for ordinal test and train
X_train_ord = X_train[ordinal_columns]
X_test_ord = X_test[ordinal_columns]
# Encoding the categorical columns

ohe = OneHotEncoder(handle_unknown='ignore', drop='first')
X_train_ohe = ohe.fit_transform(X_train_cat)
X_test_ohe = ohe.transform(X_test_cat)
columns = ohe.get_feature_names_out(input_features=X_train_cat.columns)
X_train_processed_ohe = pd.DataFrame(X_train_ohe.todense(), columns=columns)
X_test_processed_ohe = pd.DataFrame(X_test_ohe.todense(), columns=columns)
# Encoding the ordinal columns

ore = OrdinalEncoder(categories=ordinal_cols_mapping, handle_unknown='use_encoded_value', unknown_value=-1)
X_train_ore = ore.fit_transform(X_train_ord)
X_test_ore = ore.fit_transform(X_test_ord)
columns = ore.get_feature_names_out()
X_train_processed_ore = pd.DataFrame(X_train_ore, columns=columns)
X_test_processed_ore = pd.DataFrame(X_test_ore, columns=columns)
# Concatenating the processed dataframes

X_train_boruta = pd.concat([X_train_processed_ohe, X_train_processed_ore], axis=1)
X_test_boruta = pd.concat([X_test_processed_ohe, X_test_processed_ore], axis=1)

# Defining the selector object with Boruta

feat_selector = BorutaPy(model_rf, n_estimators='auto', verbose=2, random_state=42)
feat_selector.fit(X_train_boruta.values, y_train.values)

# Print accepted features as well as features that boruta did not ###deem unimportant or important (area of irresolution)
accept = X_train_boruta.columns[feat_selector.support_].to_list()
irresolution = X_train_boruta.columns[feat_selector.support_weak_].to_list()
print('Accepted features:')
print(list(accept))
print('Irresolution features:')
print(list(irresolution))

model_rf_boruta = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight='balanced',
    n_jobs=-1,
    verbose=True
)

# Training a model with just accepted features from Boruta

model_rf_boruta.fit(X_train_boruta[list(accept)], y_train)

y_pred_boruta = model_rf_boruta.predict(X_test_boruta[list(accept)])

print('Accuracy score:', recall_score(y_test, y_pred_boruta))
print('Recall score:', recall_score(y_test, y_pred_boruta, average='macro'))
print('Precision score:', precision_score(y_test, y_pred_boruta, average='macro'))
print('AUC score:', roc_auc_score(y_test, y_pred_boruta))
'''
disp = ConfusionMatrixDisplay.from_estimator(estimator=model_rf_boruta,
                                             X=X_test_boruta[list(accept)],
                                             y=y_test,
                                             normalize='true')
features = model_rf_boruta.feature_names_in_
importances = model_rf_boruta.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8,6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
'''
print(accept)

X_train_sel = X_train[
    ['Sex', 'Felt_lonely', 'Other_students_kind_and_helpful', 
    'Parents_understand_problems', 'Most_of_the_time_or_always_felt_lonely', 
    'Missed_classes_or_school_without_permission', 'Custom_Age', 'Physically_attacked', 
    'Physical_fighting', 'Close_friends', 'Miss_school_no_permission']]
X_test_sel = X_test[
    ['Sex', 'Felt_lonely', 'Other_students_kind_and_helpful', 
    'Parents_understand_problems', 'Most_of_the_time_or_always_felt_lonely', 
    'Missed_classes_or_school_without_permission', 'Custom_Age', 'Physically_attacked', 
    'Physical_fighting', 'Close_friends', 'Miss_school_no_permission']]

categorical_columns = [
    'Sex',
    'Felt_lonely',
    'Other_students_kind_and_helpful',
    'Parents_understand_problems',
    'Most_of_the_time_or_always_felt_lonely',
    'Missed_classes_or_school_without_permission'
]

ordinal_columns = [
    'Custom_Age',
    'Physically_attacked',
    'Physical_fighting',
    'Close_friends',
    'Miss_school_no_permission']
ordinal_cols_mapping = [
    ['11 years old or younger', '12 years old', '13 years old', '14 years old', '15 years old', '16 years old', '17 years old', '18 years old or older'],
    ['0 times', '1 time', '2 or 3 times', '4 or 5 times', '6 or 7 times', '8 or 9 times', '10 or 11 times', '12 or more times'],
    ['0 times', '1 time', '2 or 3 times', '4 or 5 times', '6 or 7 times', '8 or 9 times', '10 or 11 times', '12 or more times'],
    ['0', '1', '2', '3 or more'],
    ['0 days', '1 or 2 days', '3 to 5 days', '6 to 9 days', '10 or more days']
]
# Redefining pipelines

# Constructing the preprocessing pipeline

categorical_transformer = Pipeline(
    steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ]
)

ordinal_transformer = Pipeline(
    steps=[
        ('encoder', OrdinalEncoder(categories=ordinal_cols_mapping, handle_unknown='use_encoded_value', unknown_value=-1))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_columns),
        ('ord', ordinal_transformer, ordinal_columns)
    ]
)

model_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight='balanced',
    n_jobs=-1,
    verbose=True
)

model_pipeline_rf = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('rf', model_rf)
    ]
)

model_pipeline_rf.set_output(transform='pandas')

# Retraining the model with features selected by Boruta

model_pipeline_rf.fit(X_train_sel, y_train)

y_pred_final = model_pipeline_rf.predict(X_test_sel)

print('Accuracy score:', recall_score(y_test, y_pred_final))
print('Recall score:', recall_score(y_test, y_pred_final, average='macro'))
print('Precision score:', precision_score(y_test, y_pred_final, average='macro'))
print('AUC score:', roc_auc_score(y_test, y_pred_final))
'''
disp = ConfusionMatrixDisplay.from_estimator(estimator=model_pipeline_rf,
                                             X=X_test_sel,
                                             y=y_test,
                                             normalize='true')

#Model Explanation with SHAP

# Due to the fact that SHAPE doesn't support pipelines,
#let's use the dataset and model we trained from Boruta selection

X_shap = pd.concat([X_train_boruta[list(accept)],
                    X_test_boruta[list(accept)]], axis=0)


explainer = shap.Explainer(model_rf_boruta)
shap_values = explainer(X_shap)
# Explaining the class 1 (suffered from bullying)

#shap.plots.beeswarm(shap_values[:,:,1])
shap.plots.bar(shap_values[:,:,1])
'''

joblib.dump(model_pipeline_rf, 'bully.pkl')
print('pickle')
model_test = joblib.load('bully.pkl')
model_test.predict(X_test_sel[:2])
