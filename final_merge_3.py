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


#importing sklearn 
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing as prep
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score, plot_confusion_matrix, mean_squared_error

# Importing poltly
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.express as px

np.set_printoptions(precision=4)


# %% [markdown]
# 
# ### Linear  Regression ###
# 
# Here we will use Logistic Regression   
# 
# 
# 
# %%

# %%
# %%
X = df_out.loc[:, 'MolLogP': 'VSA_EState10']
Y = df_out['Binary_value']

# Data normalization
MinMax_scaler = prep.MinMaxScaler()
X[X.columns] = MinMax_scaler.fit_transform(X[X.columns])
Standard_scaler = prep.StandardScaler()
X['Ipc'] = Standard_scaler.fit_transform(df_out['Ipc'].values.reshape(-1, 1))

desc = X.describe()
desc.sort_values(by=desc.index[1], axis=1,  ascending= False,  inplace=True)
desc.head()

# %%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

lin_reg = linear_model.LinearRegression(normalize=False)
lin_reg.fit(X_train, Y_train)
# print('Coefficients: \n', lin_reg.coef_)
print('Variance score: {}'.format(lin_reg.score(X_test, y_test)))
# Calculate the absolute error
lin_reg_errors = abs(lin_reg.predict(X_test) - y_test)
print('Test Set Mean Absolute Error:', round(np.mean(lin_reg_errors), 2))


# %% [markdown]
## Ridge Regression Model
# %%
# creating a rig regression model (L2 regressor)
alphas = np.logspace(-2,0,10)
# NonZeroCoeff_L2: the number of non-zero coefficients for all alphas
# Model_Coeff_L2 : the values of all the coefficients for all alphas
Train_Error_L2 = np.zeros(shape=(len(alphas),1))
Test_Error_L2 = np.zeros(shape=(len(alphas),1))

Model_Coeff_L2  = np.zeros(shape=(x.shape[1], len(alphas)))
NonZeroCoeff_L2 = np.zeros(shape=(len(alphas),1))
count = -1
for alpha in alphas:
    count += 1
    # instantiate a logistic regression model, and fit with X and y
    model_L2 = linear_model.Ridge(alpha=alpha)
    # Fit the model
    model_L2.fit(X_train, Y_train)
    Train_Error_L2[count,0] = fit_goodness(model_L2,X_train, y_train)
    Test_Error_L2[count,0]  = fit_goodness(model_L2,X_test, y_test)
    Model_Coeff_L2[:,count] = model_L2.coef_
    NonZeroCoeff_L2[count,0] = x.shape[1] - (model_L2.coef_ == 0).sum()
plt.scatter(alphas, Test_Error_L2, label='Test')
plt.scatter(alphas, Train_Error_L2, label='Train')
plt.xlabel(r'$\alpha$')
plt.ylabel('Standard Mean Square Error')
plt.legend()
plt.show()
print('Number of Non zero Coeff: ',NonZeroCoeff_L2)

#%% [markdown]
#  TODO: check
### Ridge Regression Cross validation
#%%
# %%
splitter = KFold(5, random_state=42, shuffle=True)
alphas = np.logspace(-2,0,50)
model_RidgeCV = linear_model.RidgeCV(alphas=alphas, cv=splitter).fit(X_train,y_train)
print('Number of tested alphas: ',alphas.shape)
print('best alpha: ',round(model_RidgeCV.alpha_,3))
print('Non zero coefficients: ',x.shape[1] - (model_RidgeCV.coef_ == 0).sum())
print('Training Standard mean sqr error: ', fit_goodness(model_RidgeCV,X_train,y_train))
print('Test Standard mean sqr error: ', fit_goodness(model_RidgeCV,X_test,y_test))

# %% [markdown]
## Lasso Regression
#%%
alphas = np.logspace(-1,0,20)
print(alphas)
# creating a Lasso regression
Train_Error_L1 = np.zeros(shape=(len(alphas),1))
Test_Error_L1 = np.zeros(shape=(len(alphas),1))
Model_Coeff_L1  = np.zeros(shape=(x.shape[1], len(alphas)))
NonZeroCoeff_L1 = np.zeros(shape=(len(alphas),1))
count = -1
for alpha in alphas:
    count += 1
    # instantiate a logistic regression model, and fit with X and y
    model_L1 = linear_model.Lasso(alpha=alpha, max_iter=2000)
    # Fit the model
    model_L1.fit(X_train, y_train)
    Train_Error_L1[count,0] = fit_goodness(model_L1,X_train, y_train)
    Test_Error_L1[count,0]  = fit_goodness(model_L1,X_test, y_test)
    Model_Coeff_L1[:,count] = model_L1.coef_
    NonZeroCoeff_L1[count,0] = x.shape[1] - (model_L1.coef_ == 0).sum()

plt.scatter(alphas, Test_Error_L1, label='Test')
plt.scatter(alphas, Train_Error_L1, label='Train')
plt.title(r'Standard Mean Square Error as a function of $\alpha$')
plt.xlabel(r'$\alpha$')
plt.ylabel('Standard Mean Square Error')
plt.legend()
plt.show()

plt.scatter(alphas,NonZeroCoeff_L1, label = 'non-zero coeff')
plt.legend()
plt.title(r'Number of non-zero coefficients as a function of $\alpha$')
plt.xlabel(r'$\alpha$')
plt.ylabel('Count of non-zero coefficients')
plt.show()

print('Number of Non zero Coeff: ', NonZeroCoeff_L1)

# %% [markdown]
### Lasso regression cross validation
# %%
## Cross validation Lasso Regression
splitter = KFold(5, random_state=42, shuffle=True)
alphas = np.logspace(-1, 0, 50)
model_LassoCV = linear_model.LassoCV(alphas=alphas, cv=splitter, max_iter=10000).fit(X_train, y_train)
print('Number of tested alphas: ',alphas.shape)
print('best alpha: ',round(model_LassoCV.alpha_,3))
print('Non zero coefficients: ',x.shape[1] - (model_LassoCV.coef_ == 0).sum())
print('Training Standard mean sqr error: ', fit_goodness(model_LassoCV,X_train,y_train))
print('Test Standard mean sqr error: ', fit_goodness(model_LassoCV,X_test,y_test))


# %%