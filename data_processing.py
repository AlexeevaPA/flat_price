import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    

df = pd.read_csv("train.csv")

# Columns containing duplicate information

df = df.drop(['ID_metro',
 'ID_railroad_station_walk',
 'ID_railroad_station_avto',
 'ID_big_road1',
 'ID_big_road2',
 'ID_railroad_terminal',
 'ID_bus_terminal'], axis=1)

df = df.drop('id', axis=1)


df = df.assign(log_price_doc=np.log1p(df['price_doc'])) # take a logarithm of price to estimate MSLE
df = df.drop('price_doc', axis=1)

numeric_columns = df.loc[:,df.dtypes!=np.object].columns # quantitative features
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean()) # replace None values with mean

correlation(df, 0.9) # delete features with a correlation of more than 0.9

numeric_columns = df.loc[:,df.dtypes!=np.object].columns

cutter = VarianceThreshold(threshold=0.1) # find features with a variance of more than 0.1
cutter.fit(df[numeric_columns])
constant_cols = [x for x in numeric_columns if x not in cutter.get_feature_names_out()]

categorical_columns = df.loc[:,df.dtypes==np.object].columns

for col in categorical_columns: # implement of one-hot-encoding to change feature columns with unique values less than 5, otherwise delete them
    if col != 'timestamp': 
        if df[col].nunique() < 5:
            one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat((df.drop(col, axis=1), one_hot), axis=1)

        else:
            mean_target = df.groupby(col)['log_price_doc'].mean()
            df[col] = df[col].map(mean_target)

# modify time format representation 

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['month'] = df.timestamp.dt.month
df['year'] = df.timestamp.dt.year

# sort value to predict future price

df = df.sort_values('timestamp')

# implementation of one-hot-encoding to modify years and months columns, days won't be used

one_hot = pd.get_dummies(df['year'], prefix='year', drop_first=True)
df = pd.concat((df.drop('year', axis=1), one_hot), axis=1)
one_hot = pd.get_dummies(df['month'], prefix='month', drop_first=True)
df = pd.concat((df.drop('month', axis=1), one_hot), axis=1)
df = df.drop('timestamp', axis=1)