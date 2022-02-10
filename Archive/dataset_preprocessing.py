#%%

import numpy as np
import pandas as pd
from sklearn import preprocessing as prep
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold



# %%
# analyzing the data set to discover features with high values to scale
df = pd.read_csv('data/Unique_features.csv', encoding= 'cp1250', index_col= 'ID')
desc = df.describe()
desc.sort_values(by=desc.index[1], axis=1,  ascending=False,  inplace=True)
print(desc.head())

# %%
# Scaling using MinMax approach
MinMax_scaler = prep.MinMaxScaler()
df_sc = df.copy(deep=True)
df_sc[df_sc.columns] = MinMax_scaler.fit_transform(df_sc[df_sc.columns])
# desc_filt = desc.loc[:, (desc.loc['mean', :] >= 1)]
# print(desc_filt.head())

# desc = df_sc.describe()
# desc.sort_values(by=desc.index[1], axis=1,  ascending=False,  inplace=True)
df_sc.to_csv('data/scaled_data_1.csv')

Standard_scaler = prep.StandardScaler()
df_sc['Ipc'] = Standard_scaler.fit_transform(df_sc['Ipc'].values.reshape(-1, 1))
df_sc.to_csv('data/scaled_data_2.csv')

desc = df_sc.describe()
desc.sort_values(by=desc.index[1], axis=1,  ascending= False,  inplace=True)

print(desc.head())

# %%
# Depending on the type of analysis should be the type of scaling normalization

