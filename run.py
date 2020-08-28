import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def get_uniques(df, column_name):
    column = df.dropna(subset = [column_name])[column_name]
    splited = list(column.str.split("|", expand = False ))

    flatten = lambda l: [item for sublist in l for item in sublist]
    splited = flatten(splited)

    return np.unique(np.array(splited))  


def filter_features(features):
    
    features =list(features)
    if "both" in features:
        features.remove("both")
    
    if "no" in features and "yes" not in features:
        features.remove("no")
        
    return features


# couldn't use one-hot encoder because of multiple categories and "both" values
# This function should be applied only on 'object' dtype columns
def split_column(df, column_name):

    is_in = lambda df, col, label: [1 if (label in item) else 0 for item in df[col]]
    is_ = lambda df, col, label: [1 if (label == item or item == "both") else 0 for item in df[col]]
    
    uniques = get_uniques(df, column_name)
    type_check = (uniques == pd.Series(df[column_name]).unique())
        
    uniques = filter_features(uniques)
    
    if np.all(type_check):

        for label in uniques:
            new_col = column_name+"___"+label
            new_col = new_col.replace(' ', '_').replace('-', '_').replace(',', '_').replace('(', '').replace(')', '')

            df.insert(0, new_col, is_(df, column_name, label))
                
    else:
        for label in uniques:
            new_col = column_name+"___"+label
            new_col = new_col.replace(' ', '_').replace('-', '_').replace(',', '_').replace('(', '').replace(')', '')

            df.insert(0, new_col, is_in(df, column_name, label))
            
    df.drop(column_name, axis = 1, inplace = True)
    
    return df


#Drop the column if the majority of values are 0 for example. K is the coefficient for describing the "majority"
def drop_columns(df, k):
    for col in df.columns:
        count = max(df[col].value_counts())
        if count > k * len(df):
            df.drop(col, axis = 1, inplace = True)
            
    return df


#______LOADING DATA______

#Loading the dataset and replacing some unnecessary values with NaN

DATAPATH = 'data.csv'
NA_VALUES = ["No Info", "Not Applicable", "Nothing", "None", "none", "unknown amount"]
df = pd.read_csv(DATAPATH, na_values = NA_VALUES, encoding = "latin1")
df.sample(frac=1)

#Fixing column names: making all the letters lowercase and adding underscores instead of spaces and other symbols

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace(',', '_')



#Making all the values lowercase in the dataset, because in some column I had values such as "YES", "yes" and "Yes"

for column_name in df.select_dtypes(include=['object']).columns:
    df[column_name] = df[column_name].str.lower()


#seperate the dependent variable

COMPANY_STATUS = LabelBinarizer().fit_transform(df.dependent_company_status)
df.drop("dependent_company_status", axis = 1, inplace = True)


# _____________DATA PREPROCESSING__________________

#Dropping column with unique values more than 1/3 of dataset length

for column_name in df.select_dtypes(include=['object']).columns:
    if  len(get_uniques(df, column_name))*3 > len(df):
        #print(column_name, "column was droped, number of unique values was", len(get_uniques(df, column_name)))
        df.drop(column_name,  axis = 1,  inplace = True)


#Dropping column with NaN values more than half of dataset length

for column_name in df.columns:
    if  (df[column_name].isnull().sum())*2 > len(df):
        #print(column_name, "column was dropped, number of NaNs was", df[column_name].isnull().sum())
        df.drop(column_name,  axis = 1,  inplace = True)


# Fillig NaNs depending on the column datatypes. for objects forwardfill. for numeric fill with the mean of the column.

obj_col = df.select_dtypes(include=['object']).columns
df[obj_col] = df[obj_col].fillna(method="ffill")
df[obj_col] = df[obj_col].fillna(method="bfill") #in the first row NaN-s will still remain after the previous step

df.fillna(df.mean(), inplace = True)


#Split columns for categorical values

for column_name in df.select_dtypes(include=['object']).columns:
       
    df = split_column(df, column_name)


#dropping some columns for more see function definition

df = drop_columns(df, k = 9/10)


#Normalize all the columns of dataset - rescale from 0 to 1 for being less sensitive to the scale of features

for column_name in df.columns:
    
    normalize = lambda df, column_name, min, max: [(value - min)/(max-min) for value in df[column_name]]
    try:
        df[column_name] = normalize(df, column_name,df[column_name].min(), df[column_name].max())
    except ZeroDivisionError:
        df.drop(column_name, axis = 1, inplace = True)
        print(column_name, " column has been droped")

print("DataFrame shape - ", df.shape)
#____________________________________________________________________________________________________________________

#_________TRAINING XGBRegressor_________________

X_train, X_test, y_train, y_test = train_test_split(df, COMPANY_STATUS, test_size=0.2, random_state=42)

xg_reg = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.1, learning_rate = 0.1, max_depth = 10, alpha = 1, n_estimators = 1000)
xg_reg.fit(X_train,y_train)


#computing the accuracy of the prediction

flatten = lambda l: [item for sublist in l for item in sublist]

xg_preds = xg_reg.predict(X_test)
acc = abs(flatten(y_test) - xg_preds)
acc = [acc < 0.2]
acc = sum(sum(acc)) / len(xg_preds)*100
print (round(acc, 1), "%")


#Plotting feature importances

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [10, 5]
plt.show()

