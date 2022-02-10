#%%

import numpy as np
import pandas as pd
from sklearn import preprocessing as prep
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.express as px

#importing rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

from sklearn.model_selection import train_test_split

#%%

df = pd.read_csv('data/Unique_features.csv', encoding= 'cp1250', index_col='ID')
df_c = pd.read_csv('data/cleaned_CatS_CHEMBL.csv', index_col='ID')
df_out = df.copy(deep=True)
df_out = pd.concat([df_out, df_c['IC50']], axis =1, join= 'inner')
print(df_out.head())


# %%
treshold = 1000
cat_df = (df_out['IC50'] < treshold).apply(np.count_nonzero)
cat_df.name= 'Binary_value'

n_Inb = cat_df.sum()
print('Number of Inhibitors: {}'.format(n_Inb))
n_nonInb = (df_out['IC50'] >= treshold).apply(np.count_nonzero).sum()
print('Number of Non-Inhibitors: {}'.format(n_nonInb))
df_out = pd.concat([df_out, cat_df], axis =1, join= 'inner')
df_out.to_csv('data/features_out_cat.csv')


# %%

min_y = df_out['IC50'].min()
max_y = df_out['IC50'].max()
fig = px.histogram(df_out, x = 'IC50', range_x=[0, max_y], color= 'Binary_value' )
fig.show()

# %%
min_y = df_out['pchembl'].min()
max_y = df_out['pchembl'].max()
fig = px.histogram(df_out, x = 'pchembl', color= 'Binary_value' )

layout = dict(
            xaxis=dict(title='-LOG IC50', showline=False),
            yaxis=dict(title='Counts', showline=False),
            hovermode='closest',
            title='Distribution of the Target variable',
            title_x=0.5,
            font=dict(
                family='Open Sans',
                size=14,
                color='black'
            )
    )

fig.update_layout(layout)
pio.show(fig)

plot_url_1 = py.plot(fig, filename='Distribution_output_log', auto_open=True,) 
print ('Plot Online in {}'.format(plot_url_1))

# %%

antilog_vals = (df_out['pchembl']*-1).apply(lambda x: (10**x)/10**-9)
print(antilog_vals)

# %%

ones = df_out.loc[df_out['IC50'] < treshold,:]
zeros = df_out.loc[df_out['IC50'] >= treshold,:]

# %%
