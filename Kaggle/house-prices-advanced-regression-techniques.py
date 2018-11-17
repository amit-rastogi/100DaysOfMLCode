# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.

#function will return list of NaN columns for both numerical and string column
def listOfNaNColumns(listOfCols, df):
   cols_num = []
   cols_str = []
   for col in listOfCols:
        if df[col].dtype == object:
          cols_str.append(col)
        else:
          cols_num.append(col)     
   return cols_num, cols_str


# Read the training data and create matrix of independent features
train_X = pd.read_csv('../input/train.csv')

# Read the test data
test_X = pd.read_csv('../input/test.csv')
test_X_id = test_X.Id
#create vector of dependent variable
train_y = train_X.SalePrice

#delete the dependent column from training dataframe
del train_X['SalePrice']

#perform imputing on numeric and string columns
nan_num_cols_train, nan_str_cols_train = listOfNaNColumns(train_X.columns[train_X.isnull().any()].tolist(), train_X)
nan_num_cols_test, nan_str_cols_test = listOfNaNColumns(test_X.columns[test_X.isnull().any()].tolist(), test_X)

imputer_num = Imputer(missing_values='NaN', strategy='mean', axis=0)
for col_num in nan_num_cols_train:
    train_X[[col_num]] = imputer_num.fit_transform(train_X[[col_num]])
for col_num in nan_num_cols_test:
    test_X[[col_num]] = imputer_num.fit_transform(test_X[[col_num]])

#for object type column need to manually fill nans with the most frequent value due to Imputer limitation
for col in nan_str_cols_train:
    train_X[col].fillna(train_X[col].value_counts().idxmax(), inplace=True)
for col in nan_str_cols_test:
    test_X[col].fillna(test_X[col].value_counts().idxmax(), inplace=True) 
    
list_of_string_cols = [key for key in dict(train_X.dtypes) if dict(train_X.dtypes)[key] in ['object']]

#By data analysis we can see that GarageQual and GarageCond need custom mapping
custom_map_garage_attr = {'Ex': 6, 'Gd': 5, 'TA':4, 'Fa':3, 'Po':2, 'NA':1}
train_X['GarageQual'] = train_X['GarageQual'].map(custom_map_garage_attr)
train_X['GarageCond'] = train_X['GarageCond'].map(custom_map_garage_attr)
test_X['GarageQual'] = test_X['GarageQual'].map(custom_map_garage_attr)
test_X['GarageCond'] = test_X['GarageCond'].map(custom_map_garage_attr)

#remove the above columns from the list for labelencoding
list_of_string_cols.remove('GarageQual')
list_of_string_cols.remove('GarageCond')

#label encoding and one hot encoding of categorical features
labelenc = LabelEncoder()

for col in list_of_string_cols:
      train_X[col] = labelenc.fit_transform(train_X[col])
      test_X[col] = labelenc.fit_transform(test_X[col])
    
list_of_col_indexes = []
for col in list_of_string_cols:
    list_of_col_indexes.append(train_X.columns.get_loc(col))

onehotenc = OneHotEncoder(categorical_features = [list_of_col_indexes])
train_X =  onehotenc.fit_transform(train_X).toarray()
test_X =  onehotenc.transform(test_X).toarray()

train_X_sub, test_X_sub, train_y_sub, test_y_sub = train_test_split(train_X, train_y, test_size=0.25)

#fit XGB Boost regressor
XGBmodel = XGBRegressor(n_estimators=1000, learning_rate=0.05)
XGBmodel.fit(train_X_sub, train_y_sub, early_stopping_rounds=5, eval_set=[(test_X_sub, test_y_sub)], verbose=False)

#make prediction
prediction = XGBmodel.predict(test_X)

my_submission = pd.DataFrame({'Id': test_X_id, 'SalePrice': prediction})
# you could use any filename. We choose submission here
my_submission.to_csv('./submission.csv', index=False)