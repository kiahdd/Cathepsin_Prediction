# %%

# importing required packages
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

# importing high-level chart Objes
import plotly.graph_objects as go

# %%
# online plotly module
# import chart_studio.plotly as py
# import chart_studio
# chart_studio.tools.set_credentials_file(username='kianahaddadi', api_key='OPUsSMNwSNh1pgaA3yVF')

# %%
compounds = pd.read_csv('data/cleaned_CatS_CHEMBL.csv',index_col='ID')
compounds['molecules'] = compounds.smiles.apply(Chem.MolFromSmiles)
compounds['molecules']
# %%

# creating fingerprints
compounds['RDKfingerprints'] = [Chem.RDKFingerprint(x) for x in compounds.molecules]
compounds['MorganFingerprint'] = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in compounds.molecules]
# %%

# %%

## calculating similarity using another fingerprint
def pairwise_sim(mols):
    pairwise=[]
    for i in mols:
        sim = DataStructs.BulkTanimotoSimilarity(i,mols)
        pairwise.append(sim)
    print('Done!')
    return pairwise

# %%
PW_sim_1 = pairwise_sim(compounds['RDKfingerprints'])
PW_sim_1 = DataFrame(PW_sim_1, index = compounds.index, columns = compounds.index)
PW_sim_1.head
# %%
PW_sim_2 = pairwise_sim(compounds['MorganFingerprint'])
PW_sim_2 = DataFrame(PW_sim_2, index = compounds.index, columns = compounds.index)
PW_sim_2.head
#%%
# creating a smaller similarity matrix and testing the similarity result
matr_sim2 = PW_sim_1.loc[:,'CHEMBL1076795':'CHEMBL1079101']
# %%
# create a scatter matrix for some compounds
pd.plotting.scatter_matrix(matr_sim2, alpha=0.2, diagonal = 'kde', figsize=(12,12))

# %%
# Generate a mask for the upper triangle
matr_sim2 = PW_sim_1.loc['CHEMBL1076795':'CHEMBL1079101','CHEMBL1076795':'CHEMBL1079101']
mask = np.ones_like(matr_sim2, dtype=np.bool)
mask[np.triu_indices_from(mask)] = False
matr_sim2

# %%
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
Draw.MolsToGridImage(compounds['molecules'].loc[matr_sim2.columns],molsPerRow=4,legends=list(matr_sim2.columns))
# %%
data = [go.Histogram(x=DataFrame(PW_sim_2).stack(), nbinsx=100)]
iplot(data, filename='GetMorganFingerprintAsBitVect similarity')

data = [go.Histogram(x=DataFrame(PW_sim_1).stack(), nbinsx=100)]
iplot(data, filename='RDKfingerprint similarity')

# %%
fig = go.Figure()
fig.add_trace(go.Histogram(x=DataFrame(PW_sim_2).stack(), nbinsx=100))
fig.add_trace(go.Histogram(x=DataFrame(PW_sim_1).stack(), nbinsx=100))

# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show()
# %%
fig.write_image("figures/fingerprint_similarity_comparison.png")
# %%

PW_sim_1.to_csv('data/RDKfingerprint_similarity.csv')
PW_sim_2.to_csv('data/Morganfingerprint_similarity.csv')

# %%