# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# import os
# try:
# 	os.chdir(os.path.join(os.getcwd(), 'D:\Code\PycharmProjects\CHE1147-DM\Git\CHE1147-Project'))
# 	print(os.getcwd())
# except:
# 	pass
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


#importing sklearn 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing as prep
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score, plot_confusion_matrix

# Importing poltly
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.express as px

np.set_printoptions(precision=4)
# %% [markdown]
# ## 7. Supervised Learning - Analysis of the Output Distribution ##
# 
# In this part we will analyze the distribution of 
# the output and convert the numerical values to 
# categories according with a threshold.
# 
# We will use a threshold of 100 for the IC50 value to classify 
# the compound as a strong or a weak inhibitor.
# 

# %%
df = pd.read_csv('data/Unique_features.csv', encoding= 'cp1250', index_col='ID')
df_c = pd.read_csv('data/cleaned_CatS_CHEMBL.csv', index_col='ID')
df_out = df.copy(deep=True)
df_out = pd.concat([df_out, df_c['IC50']], axis =1, join= 'inner')


df_d3c_pca = pd.read_csv('data/D3C_pca_scores.csv', index_col= 'Cmpd_ID') 
df_d3c = pd.read_csv('data/D3C_clean_features.csv', index_col= 'Cmpd_ID')


# %%

threshold = 100
df_out['Binary_value'] = (df_out['IC50'] <= threshold).apply(np.count_nonzero)
df_d3c['Binary_value'] = (df_d3c['Affinity']*1000 <= threshold).apply(np.count_nonzero)

n_Inb = df_out['Binary_value'].sum()
print('Number of Strong Inhibitors: {}'.format(n_Inb))
n_nonInb = (df_out['IC50'] > threshold).apply(np.count_nonzero).sum()
print('Number of Weak Inhibitors: {}'.format(n_nonInb))
print('Percentage of Strong Inhibitors: {} %'.format(100*n_Inb/(n_Inb+ n_nonInb)))
print('Percentage of Weak Inhibitors: {} %'.format(100*n_nonInb/(n_Inb+ n_nonInb)))

# df_out.to_csv('data/features_out_cat.csv')
# df_d3c.to_csv('data/D3C_features_out_cat.csv')

# %% [markdown]
# ** Plotting the Output distribution **
# 
# Our Output is the IC50 value which unit is in nM. This means 
# that the value in the standard unit (M) would be X x 10 ^ -9 
# where X is the reported IC50 value in the data set. 
# 
# 
#%%
min_y = df_out['IC50'].min()
max_y = df_out['IC50'].max()
fig = px.histogram(df_out, x = 'IC50', range_x=[0, max_y], color= 'Binary_value' )
layout = dict(
            xaxis=dict(title='IC50', showline=False),
            yaxis=dict(title='Frequency', showline=False),
            hovermode='closest',
            title='Distribution of the Target variable',
            title_x=0.5,
            font=dict(
                family='Open Sans',
                size=14,
                color='black'
            ),
            annotations=list([
                dict(
                    x=1.15,
                    y=1.07,
                    xref='paper',
                    yref='paper',
                    text='Categories',
                    showarrow=False,
                )
            ])
    )
fig.update_layout(layout)
fig.show()



# %% [markdown]
# 
# 
# As we can observe the distribution of the output is not normal
#  with high frequencies of low values and just a few of very 
# high values. 
# 
# We decided to use the *-LOG* of the IC50 values, which is already calculated and reported as 'pchembl' in the data set
# 

# %%
min_y = df_out['pchembl'].min()
max_y = df_out['pchembl'].max()
fig = px.histogram(df_out, x = 'pchembl', color= 'Binary_value' )

layout = dict(
            xaxis=dict(title='-LOG IC50', showline=False),
            yaxis=dict(title='Frequency', showline=False),
            hovermode='closest',
            title='Distribution of the Target variable',
            title_x=0.5,
            font=dict(
                family='Open Sans',
                size=14,
                color='black'
            ),
           annotations=list([
                dict(
                    x=1.15,
                    y=1.07,
                    xref='paper',
                    yref='paper',
                    text='Categories',
                    showarrow=False,
                )
            ])
    )

fig.update_layout(layout)
pio.show(fig)

# %%
min_y = df_d3c['pIC50'].min()
max_y = df_d3c['pIC50'].max()
fig = px.histogram(df_d3c, x = 'pIC50', color= 'Binary_value' )

layout = dict(
            xaxis=dict(title='-LOG IC50', showline=False),
            yaxis=dict(title='Frequency', showline=False),
            hovermode='closest',
            title='Distribution of the Target variable in the Test set',
            title_x=0.5,
            font=dict(
                family='Open Sans',
                size=14,
                color='black'
            ),
           annotations=list([
                dict(
                    x=1.15,
                    y=1.07,
                    xref='paper',
                    yref='paper',
                    text='Categories',
                    showarrow=False,
                )
            ])
    )

fig.update_layout(layout)
pio.show(fig)

# first_plot_url = py.plot(fig, filename='Output_distrb_D3C', auto_open=True,)
# print ('Plot Online in {}'.format(first_plot_url))

# %% [markdown]
# 
# Using the *-LOG* value we obtained a distribution closer to 
# normal, also using a threshold of 100 for the IC50 value we
#  obtained a balanced data set. 
# 



# %% [markdown]
# 
# ## 8. Supervised Learning - Classification ##
# 
# In this section we will use some Classification methods to 
# predict if a compound would be a strong or a weak inhibitor.  
# 
# 
# 
# %%
X = df_out.loc[:, 'MolLogP': 'VSA_EState10']
Y = df_out['Binary_value']

X_d3c = df_d3c.loc[:, 'MolLogP': 'VSA_EState10']
Y_d3c = df_d3c['Binary_value']

# Data normalization
MinMax_scaler = prep.MinMaxScaler()
X[X.columns] = MinMax_scaler.fit_transform(X[X.columns])
Standard_scaler = prep.StandardScaler()
X['Ipc'] = Standard_scaler.fit_transform(df_out['Ipc'].values.reshape(-1, 1))

desc = X.describe()
desc.sort_values(by=desc.index[1], axis=1,  ascending= False,  inplace=True)
desc.head()

X_d3c[X_d3c.columns] = MinMax_scaler.fit_transform(X_d3c[X_d3c.columns])
X_d3c['Ipc'] = Standard_scaler.fit_transform(df_d3c['Ipc'].values.reshape(-1, 1))

# %% [markdown]
# 
# ### Logistic Regression ###
# 
# Here we will use Logistic Regression   
# 
# 
# 
# %%


# %%
# TODO: ADD LOG REG PART 
# TODO: 
# TODO: 
# TODO: 
# TODO: 
# TODO: 
# TODO: 
# TODO: 
# TODO: 



# %% [markdown]
# 
# ### C-Support Vector Classification ###
# 
# Here we will use C-Support Vector Classification   
# 
# 
# 
# %%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
print('Ratio number of instances in the Test/ Train: {}'.format(len(Y_test)/len(Y_train)))
print('Ratio of 1 in Train set: {}'.format(Y_train.sum()/len(Y_train)))
print('Ratio of 1 in Test set: {}'.format(Y_test.sum()/len(Y_test)))

# %%
svcClassifier = SVC(kernel = 'linear', C = 1, gamma = 'auto',  random_state= 42)
svcClassifier.fit(X_train, Y_train)

Y_pred = svcClassifier.predict(X_test)
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

# %%
title = 'Normalized confusion matrix using SVC'
disp = plot_confusion_matrix(svcClassifier, X_test, Y_test,
                                display_labels=['Weak Inhibitor', 'Strong Inhibitor'],
                                cmap=plt.cm.Blues,
                                normalize='true')
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)


plt.savefig('figures/SVC_Cmatrix.png')
plt.show()
# %%
# Using CrossValidation to measure the accuracy

scores = cross_val_score(svcClassifier, X,Y, cv = 5)
print('Scores using CrossValidation: {}'.format(scores))
# %% [markdown]
# 
# Dimensionality reduction can help to improve the accuracy of the model    
# 
# 
# 
# %%
df_pca = pd.read_csv('data/pca_scores.csv', index_col= 'ID')
df_pca.head()
X_pca = df_pca
# %%
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size = 0.3, random_state = 42)

print('Ratio number of instances in the Test/ Train: {}'.format(len(Y_test)/len(Y_train)))

print('Ratio of 1 in Train set: {}'.format(Y_train.sum()/len(Y_train)))

print('Ratio of 1 in Test set: {}'.format(Y_test.sum()/len(Y_test)))

# %%
svcClassifier = SVC(kernel = 'linear', C = 1, gamma = 'auto',  random_state= 42)
svcClassifier.fit(X_train, Y_train)

Y_pred = svcClassifier.predict(X_test)
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
print(accuracy_score(Y_test, Y_pred))
print(average_precision_score(Y_test, Y_pred))


title = 'Normalized confusion matrix using SVC and PCA'
disp = plot_confusion_matrix(svcClassifier, X_test, Y_test,
                                display_labels=['Weak Inhibitor', 'Strong Inhibitor'],
                                cmap=plt.cm.Blues,
                                normalize='true')
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)


plt.savefig('figures/SVC_Cmatrix_pca.png')
plt.show()

# %%
scores = cross_val_score(svcClassifier, X,Y, cv = 5)
scores
# %% [markdown]
# 
# ** Fine tunning of parameter for the SVC model**
# 
# Here we will use Grid Search + Cross validation to find 
# the best parameters.    
# 
# 
# %%

X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size = 0.3, random_state = 42)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 'scale', 'auto'],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}, {'kernel': ['poly'], 'gamma': [1e-3, 1e-4, 'scale', 'auto'],
                     'C': [1, 10, 100, 1000]} ]

# scores = ['precision', 'recall']
scores = ['precision']

#%%

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, Y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true,y_pred))

    title = 'Normalized confusion matrix using SVC and GridsearchCV'
    disp = plot_confusion_matrix(svcClassifier, X_test, Y_test,
                                    display_labels=['Weak Inhibitor', 'Strong Inhibitor'],
                                    cmap=plt.cm.Greens,
                                    normalize='true')
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)


    plt.savefig('figures/SVC_Cmatrix_CV_{}.png'.format(str(score)))
    plt.show()

# %% [markdown]
# 
# ### Test using the D3C data set ###   
# 
# 
# 


# %%
#  TODO:

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

# X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size = 0.3, random_state = 42)

svcClassifier = SVC(kernel = 'linear', C = 1, gamma = 'scale', degree= 5 ,  random_state= 42)
svcClassifier.fit(X_train, Y_train)

Y_pred = svcClassifier.predict(X_d3c)
print(confusion_matrix(Y_d3c,Y_pred))
print(classification_report(Y_d3c,Y_pred))
print(accuracy_score(Y_d3c, Y_pred))
print(average_precision_score(Y_d3c, Y_pred))


title = 'Normalized confusion matrix using SVC'
disp = plot_confusion_matrix(svcClassifier, X_d3c, Y_d3c,
                                display_labels=['Weak Inhibitor', 'Strong Inhibitor'],
                                cmap=plt.cm.Purples,
                                normalize='true')
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)

plt.savefig('figures/SVC_Cmatrix_D3C.png')
plt.show()

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size = 0.3, random_state = 42)

svcClassifier = SVC(kernel = 'rbf', C = 100, gamma = 0.001, degree= 5 ,  random_state= 42)
svcClassifier.fit(X_train, Y_train)

Y_pred = svcClassifier.predict(df_d3c_pca)
print(confusion_matrix(Y_d3c,Y_pred))
print(classification_report(Y_d3c,Y_pred))
print(accuracy_score(Y_d3c, Y_pred))
print(average_precision_score(Y_d3c, Y_pred))


title = 'Normalized confusion matrix using SVC and PCA'
disp = plot_confusion_matrix(svcClassifier, df_d3c_pca, Y_d3c,
                                display_labels=['Weak Inhibitor', 'Strong Inhibitor'],
                                cmap=plt.cm.Purples,
                                normalize='true')
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)

plt.savefig('figures/SVC_Cmatrix_D3C_pca.png')
plt.show()

# %% [markdown]
# 
# ### k-nearest neighbors (KNN) ###
# 
# Here we will use k-nearest neighbors (KNN) method   
# 
# 
# 
# %%
#Scaling the features using standard scaler ((x-mean)/std_dev)
# scaler = prep.StandardScaler()
# scaler.fit(X)

# X_sc = scaler.transform(X)
# %%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

classifier = KNeighborsClassifier(n_neighbors= 12)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

title = 'Normalized confusion matrix using KNN'
disp = plot_confusion_matrix(classifier, X_test, Y_test,
                                display_labels=['Weak Inhibitor', 'Strong Inhibitor'],
                                cmap=plt.cm.Blues,
                                normalize='true')
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)

plt.show()
plt.savefig('figures/KNN_Cmatrix.png', format= 'png')
# %%
error_train = []
error_test = []

# Calculating error for K values between 1 and 40
for i in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_s = knn.predict(X_train)
    pred_i = knn.predict(X_test)
    error_train.append(np.mean(pred_s != Y_train))
    error_test.append(np.mean(pred_i != Y_test))

t_test1= dict(
    type= 'scatter',
    x=list(range(1, 100)), 
    y=error_test,
    name='Test',
    mode = 'lines+markers',
    line = dict(
        dash='dot',
        color='darkgreen',
        width= 1
    ),
    marker=dict(
        color='orangered',
        symbol = 2,
        size= 5
    )
)
t_train1= dict(
    type= 'scatter',
    x=list(range(1, 100)), 
    y=error_train,
    name='Train',
    mode = 'lines+markers',
    line = dict(
        dash='dot',
        color='darkblue',
        width= 1
    ),
    marker = dict(
        # h='d',
        color='purple',
        symbol = 0,
        size= 5
    )
)
layout = dict(
    xaxis=dict(title='K value', showline=False),
    yaxis=dict(title='Mean Error', showline=False),
    hovermode='closest',
    title='Error rate vs K value',
    title_x=0.5,
    font=dict(
        family='Open Sans',
        size=14,
        color='black'
    ),
    annotations=list([
        dict(
            x=1.15,
            y=1.07,
            xref='paper',
            yref='paper',
            text='Data Set',
            showarrow=False,
        )
    ])
)
fig = dict(data=[t_train1, t_test1], layout=layout)
pio.show(fig)

# plot_url_1 = py.plot(fig, filename='Error_rate_K_value', auto_open=True,) 
# print ('Plot Online in {}'.format(plot_url_1))

# %%

classifier = KNeighborsClassifier(n_neighbors= 5)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

title = 'Normalized confusion matrix using KNN'
disp = plot_confusion_matrix(classifier, X_test, Y_test,
                                display_labels=['Weak Inhibitor', 'Strong Inhibitor'],
                                cmap=plt.cm.Blues,
                                normalize='true')
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)

plt.show()
plt.savefig('figures/KNN_Cmatrix_2.png', format= 'png')


# %% [markdown]
# 
# Dimensionality reduction can help to improve the accuracy of the model    
# 
# 
# 

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size = 0.3, random_state = 42)

classifier = KNeighborsClassifier(n_neighbors= 5)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

title = 'Normalized confusion matrix using KNN'
disp = plot_confusion_matrix(classifier, X_test, Y_test,
                                display_labels=['Weak Inhibitor', 'Strong Inhibitor'],
                                cmap=plt.cm.Oranges,
                                normalize='true')
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)

plt.show()
plt.savefig('figures/KNN_Cmatrix_pca.png', format= 'png')

# %% [markdown]
# 
# In this case did not help so much
# 
# 
# 
# %%
