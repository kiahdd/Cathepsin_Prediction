# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing as prep
from sklearn.decomposition import PCA
from FeatureGeneration import get_Descriptors


# Importing poltly
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.io as pio
import plotly.figure_factory as ff

# %%
df_d3c = pd.read_csv('data/CatS_score.csv', index_col= 'Cmpd_ID')
df_feats_d3c = get_Descriptors(df_d3c['SMILES'].values, ind=df_d3c.index)


df_feats_d3c = pd.concat([df_feats_d3c, df_d3c['Affinity'] ], axis =1, join= 'inner')
df_feats_d3c['pIC50'] = df_feats_d3c['Affinity'].apply(lambda x: np.log10(x*(10**-6))*-1)

print(df_feats_d3c.head())
df_feats_d3c.to_csv('data/D3C_features.csv')


# %%

(df_feats_d3c['pIC50']).hist()

# %%
feats = df_feats_d3c.loc[:,'MolLogP' : 'VSA_EState10']
slc_feats = feats.loc[:, column_selected]


slc_feats = pd.concat([slc_feats, df_feats_d3c['Affinity'], df_feats_d3c['pIC50'] ], axis =1, join= 'inner')

print(slc_feats.head)
print(slc_feats.shape)

slc_feats.to_csv('data/D3C_clean_features.csv')
# %%
# Normalizing the data using Standardization 
Standard_scaler = prep.StandardScaler()
D3C_sc = slc_feats.copy(deep=True)
D3C_sc.drop(['Affinity'], axis=1, inplace=True)
D3C_sc[D3C_sc.columns] = Standard_scaler.fit_transform(D3C_sc[D3C_sc.columns])

# %%
n_comp = 20
pca = PCA(n_components=n_comp, svd_solver='full')
pc_labels = list(map(lambda i: 'PC{}'.format(i), range(1,n_comp+1)))
PC_scores = pca.fit_transform(D3C_sc)
pca_scores_pd = pd.DataFrame(data = PC_scores, columns = pc_labels, index = D3C_sc.index)
print('Score matrix: \n{} \n...'.format(pca_scores_pd.head()))
loadings_pd = pd.DataFrame(data = pca.components_.T, columns = pc_labels, index = D3C_sc.columns)
print('Loadings matrix: \n{} \n...'.format(loadings_pd.head()))

pca_scores_pd.to_csv('data/D3C_pca_scores.csv')
loadings_pd.to_csv('data/D3C_loadings_pd.csv')


# %%
