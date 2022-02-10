#%%
import pandas as pd
import sklearn
from sklearn.feature_selection import VarianceThreshold

# %%

#remove IDs and unncessary information
feature_data = pd.read_csv('data/data_features_1.csv')
feature_data.set_index('ID', inplace = True)
feature_data.drop(columns = ['Original Index', 'smiles', 'type'], inplace = True)
feature_data.head()

# %%
#Take only the features
features = feature_data.iloc[:, 2:]
features.shape


# %%
#Remove columns with same values in all the rows

selector = VarianceThreshold()
selected_features = selector.fit_transform(features)
column_selected = selector.get_support()

# %%

Unique_features = features.loc[:, column_selected]
Unique_features.to_csv('data/Unique_features.csv')

# %%
