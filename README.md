# West-Nile-virus-prediction
Kaggle challenge - Predict West Nile virus in mosquitos across the city of Chicago


# Project description

Given weather, location, and spraying data, this competition asked kagglers to predict when and where different species of mosquitoes will test positive for West Nile virus. The evaluation metric for prediction accuracy was ROC AUC. The data set included 8 years: 2007 to 2014. Years 2007, 2009, 2011 and 2013 were used for training while remaining four years were included in test data.

# Feature engineering

  1. Created lag features from weather data
  2. Created stacked features by using number of mosquitoes per grouping of date-trap-species in train data to generate meta-features by        constructing out-of-fold predictions for log(number of mosquitoes)
  3. Fixed leakge and created number of duplicate row feature
  4. Processed date into year, month and day

# Model description

out-of-fold predictions of log(number of mosquitoes) from 3 different regression models were used to generate meta-features and add them to the data, creating a stacked dataset. They are:
1. Random Forest (Sklearn)
2. Ridge Regression (Sklearn)
3. Lasso (Sklearn)

Classification models are:
1. XGBoost
2. RandomForest

# Dependencies

Python 2.7,
Pandas (any relatively recent version would work),
Numpy (any relatively recent version would work),
Sklearn (any relatively recent version would work),
XGBoost,
RandomForest,

