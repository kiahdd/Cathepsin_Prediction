# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


#importing rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.Chem import Draw
from FeatureGeneration import get_Descriptors
#from utility import FeatureGenerator

#importing sklearn 

import sklearn
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing as prep
from sklearn.decomposition import PCA

# Importing poltly
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.io as pio
import plotly.figure_factory as ff

# %% [markdown]
# 
# ## 1. Understanding the Raw Data ## 
# 
# The data to analyze was obtained from the CHEMBL
# database  https://www.ebi.ac.uk/chembl/ using the
# Oficial Python client library 
# https://github.com/chembl/chembl_webresource_client 
# 
# We extracted the binding activity data of all the
# compounds reported for CatS  
# 
# 
# 
# %%
# Reading the raw data extracted from CHEMBL
df_raw = pd.read_csv('data/CatS_CHEMBL.csv')
print('Shape of the dataset : {}'.format(df_raw.shape))
print('Preview of the dataset : \n{}'.format(df_raw.head()))
print('This data set is a mixture of multiple types of assays')
# %%
g_data = df_raw[['molecule_chembl_id', 'type']].groupby(['type']).agg(['count']).rename(columns={'count': 'Counts_types'})
# print(g_data / g_data.sum())
g_data['Percentage'] = g_data / g_data.sum()

print('Counts of the assay types : \n{}'.format(g_data.to_string()))

# %% [markdown]
#  ## 2. Data Cleaning ## 
# 
# In this part we will remove non-relevant columns,
# removing missing data and duplicates.
# 
# Since the IC50 assay type represent more than the 
# 70% of the data set, we decided to keep only the
#  data of the IC50 assay type.
# 
# The raw dataset only include 5 valuable columns:   

# %%
# Create new dataframe with useful columns

useful_data = {'smiles': df_raw['canonical_smiles'], 'IC50': df_raw['standard_value'], 'pchembl': df_raw['pchembl_value'], 'type': df_raw['standard_type'], 'ID' : df_raw['molecule_chembl_id'] }

df_raw_short  = pd.DataFrame(useful_data)

# Remove all rows that doesn't correspond to IC50

df_cln = df_raw_short[df_raw_short['type'] == 'IC50']

df_cln.dropna(inplace=True) # Removing missing data 
df_cln.sort_values(by=['ID'], inplace = True)
print('Preview of cleaned data: \n{}'.format(df_cln.head()))

# %%
# Remove duplicates using CHEMBL ID

df_cln.drop_duplicates(subset = 'ID', keep = 'first', inplace = True)
df_cln.set_index('ID', inplace = True)
df_cln.shape
# %% [markdown]
#  ## 3. Feature extraction using the RDkit Library  ## 
# 
# In the data set we don't have any features, but we have
#  the SMILES representation of each compound, which we can
#  use to extract features describing the structure and 
# composition of each compound.

# %%
features = get_Descriptors(df_cln.smiles.values, df_cln.index)
features = features.loc[:, 'MolLogP':]
print(features.head())

# %% [markdown]
#  ##  4. Feature selection   ## 
# 
# We want to remove those features with high correlation between others or that are duplicates.
# 
# We will use Variance Threshold for this task. 
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# %%
selector = VarianceThreshold()
selected_features = selector.fit_transform(features)
column_selected = selector.get_support()
Unique_feats = features.loc[:, column_selected]
print('Previous Features Dataframe shape : {}'.format(features.shape))
print('New Features Dataframe shape : {}'.format(Unique_feats.shape))

# %%

df_feats_out = pd.concat([Unique_feats, df_cln['IC50'], df_cln['pchembl'], df_cln['smiles'] ], axis =1, join= 'inner')
print('Complete Dataframe shape : {}'.format(df_feats_out.shape))
print(df_feats_out.head())
df_feats_out.to_csv('data/cleaned_data_features.csv')

# %% [markdown]
#  ##  5. Data set Exploratory Analysis   ## 
# 
#In order to determined the distribution of our data set, we 
# calculated the similarity among the compounds using the 
# RDkit Library to compute Fingerprints that allow us to 
# calculate the similarity.
#  
# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# %%
df_fp = df_feats_out.copy(deep=True)
df_fp = df_fp.iloc[: , -2:]
df_fp['molecules'] = df_fp.smiles.apply(Chem.MolFromSmiles)
# creating fingerprints
df_fp['RDKfingerprints'] = [Chem.RDKFingerprint(x) for x in df_fp.molecules]
df_fp['MorganFingerprint'] = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in df_fp.molecules]

print('Preview of Fingerprint DataFrame: \n{}'.format(df_fp.head()))


# %%
def pairwise_sim(mols):
    ''' function to calculate the pairwise similarity  
    '''
    pairwise=[]
    for i in mols:
        sim = DataStructs.BulkTanimotoSimilarity(i,mols)
        pairwise.append(sim)
    print('Done!')
    return pairwise

# %%

PW_sim_1 = pairwise_sim(df_fp['RDKfingerprints'])
PW_sim_1 = pd.DataFrame(PW_sim_1, index = df_fp.index, columns = df_fp.index)
print('Preview of PW_sim_1 DF: \n{}'.format(PW_sim_1.head()))


PW_sim_2 = pairwise_sim(df_fp['MorganFingerprint'])
PW_sim_2 = pd.DataFrame(PW_sim_2, index = df_fp.index, columns = df_fp.index)
print('Preview of PW_sim_2 DF: \n{}'.format(PW_sim_2.head()))

# %% [markdown]
# 
# We will make an analysis of a small sample of the data set as demonstration and visualization of the similarity among compounds 
# 
# 
# 
# %%

# creating a smaller similarity matrix and testing the similarity result
matr_sim2 = PW_sim_1.loc[:,'CHEMBL1076795':'CHEMBL1079101']

# create a scatter matrix for some compounds
axs = pd.plotting.scatter_matrix(matr_sim2, alpha=0.2, diagonal = 'kde', figsize=(12,12))

n = len(matr_sim2.columns)
for x in range(n):
    for y in range(n):
        # to get the axis of subplots
        ax = axs[x, y]
        # to make x axis name vertical  
        ax.xaxis.label.set_rotation(45)
        # to make y axis name horizontal 
        ax.yaxis.label.set_rotation(0)
        # to make sure y axis names are outside the plot area
        ax.yaxis.labelpad = 50

# %%
# Generate a mask for the upper triangle
matr_sim2 = PW_sim_1.loc['CHEMBL1076795':'CHEMBL1079101','CHEMBL1076795':'CHEMBL1079101']
mask = np.ones_like(matr_sim2, dtype=np.bool)
mask[np.tril_indices_from(mask)] = False

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(matr_sim2, mask=mask, cmap=cmap, vmax=1, vmin=0)
plt.show()
# %%
# clustering the result to check the accuracy of values in matrix
sns.clustermap(matr_sim2, cmap = cmap)
# %%

## molecules that has been used in the clustering
Draw.MolsToGridImage(df_fp['molecules'].loc[matr_sim2.columns],molsPerRow=4,legends=list(matr_sim2.columns))

# %% [markdown]
#  
# ** Similiarity distribution of the whole data set **
# 
# 


# %%
mask1 = np.ones_like(PW_sim_1, dtype=np.bool)
mask1[np.tril_indices_from(mask1)] = False
mx1 = np.ma.masked_array(PW_sim_1, mask1)
mask2 = np.ones_like(PW_sim_2, dtype=np.bool)
mask2[np.tril_indices_from(mask2)] = False
mx2 = np.ma.masked_array(PW_sim_2, mask2)

figh = go.Figure()
figh.add_trace(go.Histogram(x=pd.DataFrame(mx2).stack(), nbinsx=51, name= 'MorganFP', marker = dict(color = 'green')))
figh.add_trace(go.Histogram(x=pd.DataFrame(mx1).stack(), nbinsx=51, name= 'RDkitFP', marker = dict(color = 'orangered')))

layout=dict(
    title='Compound Similarity Distribution',
    xaxis=dict(
        title='Similarity index'
    ),
    yaxis=dict(
        title='Frequency'
    ),
    title_x=0.5,
    barmode='overlay',
    # bargap=0,
    annotations=list([
        dict(
            x=1.25,
            y=1.07,
            xref='paper',
            yref='paper',
            text='Fingerprint Algorithm',
            showarrow=False,
        )
    ])
)

# Overlay both histograms
figh.update_layout(layout)
# Reduce opacity to see both histograms
figh.update_traces(opacity=0.75)
figh.show()

pio.write_html(figh, "figures/Similarity_histogram.html")

# %% [markdown]
#  
#  ## 6. Principal Component Analysis ## 
# 
# 
# %%
# Normalizing the data using Standardization 
Standard_scaler = prep.StandardScaler()
df_sc = df_feats_out.copy(deep=True)
df_sc.drop(['IC50','smiles'], axis=1, inplace=True)
df_sc[df_sc.columns] = Standard_scaler.fit_transform(df_sc[df_sc.columns])
# df_sc.to_csv('data/scaled_data_std.csv')

# Describing the data set to see the mean and std of each feature
desc_2 = df_sc.describe()
desc_2.sort_values(by=desc_2.index[1], axis=1,  ascending=False,  inplace=True)
print(desc_2.head())

# %%

# Performing the PCA, run test 
pca = PCA(n_components=3, svd_solver='full')
PC_scores = pca.fit_transform(df_sc)
pca_scores_pd = pd.DataFrame(data = PC_scores, columns = ['PC1', 'PC2', 'PC3'], index = df_sc.index)
print('Score matrix: \n{} \n...'.format(pca_scores_pd.head()))
loadings_pd = pd.DataFrame(data = pca.components_.T, columns = ['PC1', 'PC2', 'PC3'], index = df_sc.columns)
print('Loadings matrix: \n{} \n...'.format(loadings_pd.head()))
# %%

def expvar_plot(pca_obj):
    
    tot = sum(pca_obj.explained_variance_)
    var_exp = [(i / tot)*100 for i in pca_obj.explained_variance_]
    cum_var_exp = np.cumsum(var_exp)
    trace1 = dict(
        type='bar',
        x=['PC %s' %i for i in range(1,len(pca_obj.explained_variance_))],
        y=var_exp,
        name='Individual'
    )
    trace2 = dict(
        type='scatter',
        x=['PC %s' %i for i in range(1,len(pca_obj.explained_variance_))], 
        y=cum_var_exp,
        name='Cumulative'
    )
    data = [trace1, trace2]
    layout=dict(
        title='Explained variance by different principal components',
        title_x = 0.5,
        yaxis=dict(
            title='Explained variance in percent'
        ),
        annotations=list([
            dict(
                x=1.20,
                y=1.07,
                xref='paper',
                yref='paper',
                text='Explained Variance',
                showarrow=False,
            )
        ])
    )
    fig = dict(data=data, layout=layout)
    pio.show(fig)
    return fig

# Choosing the number of components
pca = PCA().fit(df_sc)
plot = expvar_plot(pca)
pio.write_html(plot, "figures/Explained_Variance_vs_PCs.html")
first_plot_url = py.plot(plot, filename='Explained_Variance_vs_PCs', auto_open=True,)
print ('Plot Online in {}'.format(first_plot_url))
# %% [markdown]
#  
# According with the Elbow method and the amount of the explained variance we decide to choose 20 PC's since they can explain above 90% of the variance
# 
# 

# %%
# Run PCA with the chosen number of components
n_comp = 20
pca = PCA(n_components=n_comp, svd_solver='full')
pc_labels = list(map(lambda i: 'PC{}'.format(i), range(1,n_comp+1)))
PC_scores = pca.fit_transform(df_sc)
pca_scores_pd = pd.DataFrame(data = PC_scores, columns = pc_labels, index = df_sc.index)
print('Score matrix: \n{} \n...'.format(pca_scores_pd.head()))
loadings_pd = pd.DataFrame(data = pca.components_.T, columns = pc_labels, index = df_sc.columns)
print('Loadings matrix: \n{} \n...'.format(loadings_pd.head()))

# pca_scores_pd.to_csv('data/pca_scores.csv')
# loadings_pd.to_csv('data/loadings_pd.csv')

# %%
def pc_plot(scores,loadings,loading_labels=None,score_labels=None):
    
    # adjusting the scores to fit in (-1,1)
    xt = scores[:,0]
    yt = scores[:,1]
    n = loadings.shape[0]
    scalext = 1.0/(xt.max() - xt.min())
    scaleyt = 1.0/(yt.max() - yt.min())
    xt_scaled = xt * scalext
    yt_scaled = yt * scaleyt
    # adjusting the loadings to fit in (-1,1)
    p_scaled = prep.MaxAbsScaler().fit_transform(loadings)


    trace = dict(
            type='scatter',
            x=xt_scaled,
            y=yt_scaled,
            mode='markers',
            hovertext=score_labels,
            name='Scores',
            marker=dict(
                color='#0D76BF',
                size=10,
                line=dict(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5),
                opacity=0.8)
        )
    
    layout = dict(
            xaxis=dict(title='PC1', showline=False),
            yaxis=dict(title='PC2', showline=False),
            hovermode='closest',
            title='Visualization of the data transformed by PCA',
            title_x=0.5,
            font=dict(
                family='Open Sans',
                size=12,
                color='black'
            )
    )
    o = np.zeros((p_scaled.shape[0],1))
    y_labels = loading_labels.values

    fig = ff.create_quiver(o,o, p_scaled[:,0], p_scaled[:,1],  opacity=0.8, name='Loadings', scale=1, arrow_scale=.1, line=dict(width=1, color = 'orangered'), hoverinfo='text') 
    y_text = dict(
            type='scatter',
            x=p_scaled[:,0],
            y=p_scaled[:,1],
            mode='markers',
            name='Loadings',
            hovertext=y_labels,
            marker=dict(
                color='orangered',
                size=1,
                line=None,
                opacity=0),
            showlegend = False
        )
    # print(y_labels)
    fig.add_traces([trace, y_text])
    fig.update_layout(layout)

    pio.show(fig)
    return fig


plt_pc = pc_plot(PC_scores[:,:2],loadings_pd.iloc[:,:2],loading_labels=loadings_pd.index,score_labels=pca_scores_pd.index)
pio.write_html(plt_pc, "figures/PCs_plot.html")
# %% [markdown]
#  
# Using the visualization of the PCA scores we can
#  identify some outliers. However, most of the data is 
# well distributed forming an only cluster. We can 
# identify that similar features are correlated as 
# expected.  
# 
#  We will show pictures of the chemical compounds that 
# were identified as outliers to try to understand their
#  difference with the rest
# 


# %%
# Identifying outliers

outliers_ID = ['CHEMBL567893', 'CHEMBL262697', 'CHEMBL397972', 'CHEMBL230478', 'CHEMBL2312664', 'CHEMBL2313001', 'CHEMBL3671361', 'CHEMBL3671345' ]
df_filt = df_feats_out.loc[df_feats_out.index.isin(outliers_ID) , : ]
for ind , vals in enumerate(outliers_ID):
    t_df = df_filt.loc[df_filt.index == vals , : ]
    molecules = t_df.smiles.apply(Chem.MolFromSmiles)
    if ind == 0:
        print('Group A: ')
    elif ind > 0 and ind <= 3:
        print('Group B: ')
    elif ind > 3 and ind <= 5:
        print('Group C: ')
    else:
        print('Group D: ')
    # Draw.MolsToGridImage(molecules,molsPerRow=4)
    print('Outlier ID: {}'.format(vals))
    display(Draw.MolsToImage(molecules))
# print(df_filt['ID'].values)

# %% [markdown]
#  
# As we can observe from the images, the top outlier 
# (CHEMBL567893) has a complex structure that is quite
#  different to the others outliers. Among the outliers
#  we can identify a couple of similar  structures, 
# which correspond to closer molecules.   
# 
# 
# 

# %%




