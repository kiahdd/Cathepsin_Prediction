#%%

import numpy as np
import pandas as pd
from sklearn import preprocessing as prep
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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

#%%

df = pd.read_csv('data/features_out_cat.csv', encoding= 'cp1250', index_col='ID')
df.drop(columns=['pchembl'], inplace= True)
features = df.iloc[:,:-2]
outputs = df.iloc[:,-2:]
print(features.head())



# %%
X_train, X_test, y_train, y_test = train_test_split(features, outputs['Binary_value'], test_size=0.2, random_state=0)
print(X_train.shape)
print('Train set fraction of whole data set: {}'.format(X_train.shape[0]/features.shape[0]))

# %%

lda = LDA(solver='svd' )
# train using the 80% of the data set randomly selected
X_train_lda = lda.fit_transform(X_train, y_train)
# test against whole data set
X_test_lda = lda.transform(features)


print(X_train.shape)
print(X_train_lda.shape)


# %%
def lda_plot(scores, output, ):
    zeros = np.array([i for i,x in zip(scores,output) if x==0])
    ones = np.array([i for i,x in zip(scores,output) if x==1])
    min_x= min(scores)
    max_x= max(scores)

    trace1 = dict(
        type='scatter',
        x=zeros, 
        y=np.zeros((len(zeros))),
        mode='markers',
        marker=dict(
                color='orangered',
                size=10,
                line=dict(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5),
                opacity=0.8
                ),
        name='0'
    )
    trace2 = dict(
        type='scatter',
        x=ones, 
        y=np.ones((len(ones))),
        mode='markers',
        marker=dict(
                color='darkcyan',
                size=10,
                line=dict(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5),
                opacity=0.8
                ),
        name='1'
    )
    data = [trace1, trace2]

    layout=dict(
        title='Transformed data using Linear Discriminant Analysis',
        xaxis=dict(
            title='Score',
            range=[min_x,max_x],
            # type="log"
        ),
        annotations=list([
            dict(
                x=1.22,
                y=1.05,
                xref='paper',
                yref='paper',
                text='Category',
                showarrow=False,
            )
        ])
    )

    fig = dict(data=data, layout=layout)
    pio.show(fig)
    return fig

x = np.array([a[0] for a in X_train_lda])
plot = lda_plot(x, y_train.values )

# %%
weights = lda.coef_[0]
features_id = features.columns.values
ind_max = np.where(weights == weights.max())
print(ind_max)
print(features_id[ind_max])

# %%
