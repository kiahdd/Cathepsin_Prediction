#%%

import numpy as np
import pandas as pd
from sklearn import preprocessing as prep
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.io as pio
import plotly.figure_factory as ff

#importing rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw


# %%
# Analyzing the data set to normalize 
df = pd.read_csv('data/Unique_features.csv', encoding= 'cp1250', index_col='ID')
desc = df.describe()
desc.sort_values(by=desc.index[1], axis=1,  ascending=False,  inplace=True)
print(desc.head())

# %%
# Normalizing the data using Standardization 
Standard_scaler = prep.StandardScaler()
df_sc = df.copy(deep=True)
df_sc[df_sc.columns] = Standard_scaler.fit_transform(df_sc[df_sc.columns])
df_sc.to_csv('data/scaled_data_std.csv')
desc_2 = df_sc.describe()
desc_2.sort_values(by=desc_2.index[1], axis=1,  ascending=False,  inplace=True)
print(desc_2.head())

# %%
# Performing the PCA 
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
                y=1.05,
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
# %%
# Testing the info provided by the number of PCs in the analysis

plot = expvar_plot(pca)

# %%
# Choosing the number of components
pca = PCA().fit(df_sc)
plot = expvar_plot(pca)
pio.write_html(plot, "figures/Explained_Variance_vs_PCs.html")

first_plot_url = py.plot(plot, filename='Explained_Variance_vs_PCs', auto_open=True,)
print ('Plot Online in {}'.format(first_plot_url))

# %%
# Chosen number of components
n_comp = 20
pca = PCA(n_components=n_comp, svd_solver='full')
pc_labels = list(map(lambda i: 'PC{}'.format(i), range(1,n_comp+1)))
PC_scores = pca.fit_transform(df_sc)
pca_scores_pd = pd.DataFrame(data = PC_scores, columns = pc_labels, index = df_sc.index)
print('Score matrix: \n{} \n...'.format(pca_scores_pd.head()))
loadings_pd = pd.DataFrame(data = pca.components_.T, columns = pc_labels, index = df_sc.columns)
print('Loadings matrix: \n{} \n...'.format(loadings_pd.head()))

pca_scores_pd.to_csv('data/pca_scores.csv')
loadings_pd.to_csv('data/loadings_pd.csv')

# scores_2 = pca_scores_pd.copy(deep=True)
# scores_2.sort_values(by=scores_2.columns[0], axis=0,  ascending=False,  inplace=True)
# print(scores_2.head())


# %%
# Testing the info provided by the number of PCs in the analysis

# TODO: Question: what is the implication of having 20 PCs, high redundacy or variation?
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

plot_url_2 = py.plot(plt_pc, filename='PCs_plot', auto_open=True,)
print ('Plot Online in {}'.format(plot_url_2))

# %%
# Identifying outliers
df_c = pd.read_csv('data/cleaned_CatS_CHEMBL.csv', index_col='ID')
outliers_ID = ['CHEMBL567893', 'CHEMBL262697', 'CHEMBL397972', 'CHEMBL230478', 'CHEMBL2312664', 'CHEMBL2313001', 'CHEMBL3671361', 'CHEMBL3671345' ]
df_filt = df_c.loc[df_c.index.isin(outliers_ID) , : ]
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

# %%
# Running PCA with the output column
df_out = df.copy(deep=True)
df_out = pd.concat([df_out, df_c['IC50']], axis =1, join= 'inner')
print(df_out.head())

df_out[df_out.columns] = Standard_scaler.fit_transform(df_out[df_out.columns])
df_out.to_csv('data/scaled_data_std_out.csv')
desc_3 = df_out.describe()
desc_3.sort_values(by=desc_3.index[1], axis=1,  ascending=False,  inplace=True)
print(desc_3.head())



# %%
pca_out = PCA().fit(df_out)
plot = expvar_plot(pca_out)
pio.write_html(plot, "figures/Explained_Variance_vs_PCs_out.html")

plot_url_3 = py.plot(plot, filename='Explained_Variance_vs_PCs_out', auto_open=True,)
print ('Plot Online in {}'.format(plot_url_3))

# %%
n_comp = 20
pca_out = PCA(n_components=n_comp, svd_solver='full')
pc_labels = list(map(lambda i: 'PC{}'.format(i), range(1,n_comp+1)))
PC_scores_out = pca_out.fit_transform(df_out)
pca_scores_pd_out = pd.DataFrame(data = PC_scores, columns = pc_labels, index = df_out.index)
print('Score matrix: \n{} \n...'.format(pca_scores_pd_out.head()))
loadings_pd_out = pd.DataFrame(data = pca_out.components_.T, columns = pc_labels, index = df_out.columns)
print('Loadings matrix: \n{} \n...'.format(loadings_pd_out.head()))

pca_scores_pd_out.to_csv('data/pca_scores_pd_out.csv')
loadings_pd_out.to_csv('data/loadings_pd_out.csv')

# %%
plt_pc = pc_plot(PC_scores_out[:,:2],loadings_pd_out.iloc[:,:2],loading_labels=loadings_pd_out.index,score_labels=pca_scores_pd_out.index)
pio.write_html(plt_pc, "figures/PCs_plot_out.html")

plot_url_4 = py.plot(plt_pc, filename='PCs_plot_out', auto_open=True,) 
print ('Plot Online in {}'.format(plot_url_4))

# Including the output in the PCA analysis don´t change the overall distribution and the scores of the previous PCA 

# %%
