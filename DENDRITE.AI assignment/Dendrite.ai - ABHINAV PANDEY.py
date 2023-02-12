#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 08:32:50 2023

@author: abhinavpandey
"""
#importing required library
import json

# Load the JSON data from file
path = '/Users/abhinavpandey/Downloads/Screening Test - DS/algoparams_from_ui.json'
with open(path, "r") as file:
    algo_params = json.loads(file.read())

print(algo_params)

####    QU.1) Read the target and type of regression to be run  ######

target = algo_params['design_state_data']['target']
regression_type = algo_params["design_state_data"]["target"]["type"]

print(target)
print(regression_type)


####    QU.2) Read the features (which are column names in the csv) and figure out what missing imputation needs to be applied and apply that to the columns loaded in a dataframe

#importing required library
import pandas as pd

df = pd.read_csv('/Users/abhinavpandey/Downloads/Screening Test - DS/iris.csv')

# Define a function for imputing missing values
def impute_missing_values(df, feature_name, impute_with, impute_value):
    if impute_with == "Average of values":
        imputed_value = df[feature_name].mean()
    elif impute_with == "Custom":
        imputed_value = impute_value
    else:
        imputed_value = None
    df[feature_name] = df[feature_name].fillna(imputed_value)
    return df
 
# Iterate over each feature in the algo_params                           
for feature in algo_params['design_state_data']['feature_handling']:
    if feature[1] == True:
        feature_name = feature[0]
        feature_type = feature[2]
        feature_details = feature[3]
        
        # Handle missing values for numerical features
        if feature_type == "numerical":            
            df = impute_missing_values(df, feature_name, feature_details['impute_with'], 
                                       feature_details['impute_value'])
        
        # Handle missing values for categorical features
        elif feature_type == "text":           
            df = impute_missing_values(df, feature_name, feature_details['impute_with'], 
                                       feature_details['impute_value'])
            
print(df) 


####    QU.3) Compute feature reduction based on input. See the screenshot below where there can be No Reduction, 
#Corr with Target, Tree-based, PCA. Please make sure you write code so that all options can work. 
#If we rerun your code with a different Json it should work if we switch No Reduction to say PCA.  ####


def feature_reduction():
    #define reduction_type 
    reduction_method = algo_params['design_state_data']['feature_reduction']
    num_features = 0
    
    if reduction_method == "No Reduction":
        pass    # Do nothing, all features are kept
        
    if reduction_method['feature_reduction_method'] == "Correlation with target":
        num_features = reduction_method['num_of_features_to_keep'] # number of features to keep
        
        print('num_features: ',num_features)
    elif reduction_method['feature_reduction_method'] == "Tree-based":
        # Get the number of features to keep and the tree hyperparameters
        num_features = reduction_method['num_of_features_to_keep']
        depth = reduction_method['depth_of_trees']
        num_trees = reduction_method['num_of_trees']
        
        print({'num_features':num_features,'depth':depth,'num_trees':num_trees}) 
    elif reduction_method['feature_reduction_method'] == "Principal Component Analysis":
        # Get the number of features to keep
        num_features = reduction_method['num_of_features_to_keep']
        print('num_features:',num_features)
        


print(feature_reduction())   
        
####    QU.4) Parse the Json and make the model objects (using sklean) that can handle what is required in the 
#“prediction_type” specified in the JSON (See #1 where “prediction_type” is specified). 
#Keep in mind not to pick models that don’t apply for the prediction_type specified.    ####
 

#importing required library   
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor


def model():
    prediction_type = algo_params['design_state_data']['target']['prediction_type']
    algo = algo_params['design_state_data']['algorithms'].keys()
    
    if prediction_type == 'Regression':
        for algo in ['RandomForestRegressor', 'GBTRegressor', 'LinearRegression',
                     'RidgeRegression', 'LassoRegression','SGD']:
            if algo_params['design_state_data']['algorithms'][algo]['is_selected'] == True:
                model_data = algo_params['design_state_data']['algorithms'][algo]
                print(model_data)
            
    elif prediction_type == 'Classification':
        for algo in ['RandomForestClassifier','LogisticRegression','GBTClassifier', 'DecisionTreeClassifier',
                     'SVM','KNN']:
            if algo_params['design_state_data']['algorithms'][algo]['is_selected'] == True:
                model_data = algo_params['design_state_data']['algorithms'][algo]
                print(model_data)
        
print(model())


####    QU.5) Run the fit and predict on each model – keep in mind that you need to do hyper parameter tuning 
#i.e., use GridSearchCV    ####


from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(param_grid=model(),
                    n_jobs=algo_params['hyperparameters']['parallelism'],
                    verbose=1,
                    scoring='neg_mean_absolute_error',
                    refit=True,
                    return_train_score=True,
                    max_iter=algo_params['hyperparameters']['max_iterations'],
                    n_iter=algo_params['hyperparameters']['max_search_time'])

# Fit the model on training data
grid.fit(X_train, y_train)

# Use the model to predict on test data
y_pred = grid.predict(X_test)


####    QU.6) Log to the console the standard model metrics that apply   #####

                
metric = algo_params['design_state_data']['metrics']            
print(metric)                





# =============================================================================
# AUTHOR
# ~ABHINAV PANDEY #                
# =============================================================================
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                