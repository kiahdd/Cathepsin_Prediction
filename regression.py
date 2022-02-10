# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import chart_studio
chart_studio.tools.set_credentials_file(username='kianahaddadi', api_key='DImAE8zG2VzsH0mFuTio')
import chart_studio.plotly as py
import plotly.graph_objects as go
# %% [markdown]
## Functions
# %%
def read_ChEMBL():
    compounds = pd.read_csv('data/Unique_features.csv', index_col ='ID')
    print(compounds.head())
    print('non-null values:', compounds.info())  # non-null is very important
    print('existance of null values:', compounds.isnull().sum()) 
    x = compounds.drop(['pchembl'],axis = 1)
    y = compounds['pchembl']
    print('Data shape', x.shape)
    print('target shape', y.shape)
    return x,y
    
def read_D3C():
    D3C = pd.read_csv('data/D3C_clean_features.csv', index_col='Cmpd_ID') 
    print(D3C.head())
    print('non-null values:', D3C.info())  # non-null is very important
    print('existance of null values:', D3C.isnull().sum())

    # defining target and data
    x = D3C.drop(['pIC50','Affinity'],axis = 1)
    y = D3C['pIC50']
    print('Data shape', x.shape)
    print('target shape', y.shape)
    return x,y


def Kernel_Density_Estimator(Label):
    f, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Left plot: Histogram
    sns.distplot(Label, kde=False, norm_hist=False, rug=True, label="Output: y", ax=axes[0])
    axes[0].set_title('Distribution of DFT errors: histogram')
    axes[0].set_xlabel('DFT error (kcal/mol)')
    axes[0].set_ylabel('Count of observations')

    # Right plot: Kernel Density Estimation
    sns.kdeplot(Label, shade=True, clip=(Label.min(),Label.max()), label="Output: y", ax=axes[1])
    axes[1].set_title('Kernel density estimation')
    axes[1].set_xlabel('DFT error (kcal/mol)')
    axes[0].set_ylabel('Density of observations')
    plt.legend()
    plt.show()

def Data_Splitting(feature,Label, ratio):
    X_train, X_test, y_train, y_test = train_test_split(feature,Label,test_size = ratio, random_state = 42 )
    print('Training Features Shape:', X_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Features Shape:', X_test.shape)
    print('Testing Labels Shape:', y_test.shape)
    f, axes = plt.subplots(1, 1, figsize=(12, 12))
    # Predict and Test: Kernel Density Estimation
    sns.kdeplot(y_test, shade=True, clip=(y.min(),y.max()), label="Train")
    sns.kdeplot(y_train, shade=True, clip=(y.min(),y.max()), label="Test")
    axes.set_title('Kernel density estimation')
    axes.set_xlabel('DFT error (kcal/mol)')
    axes.set_ylabel('Density of observations')
    plt.legend()
    plt.show()
    return X_train, X_test, y_train, y_test

def fit_goodness(model, test_feature, test_label):
    prediction = model.predict(test_feature)
    mse = mean_squared_error(test_label,prediction)
    rmse = round(np.sqrt(mse),3)
    return rmse

def model_accuracy(model,train_feature,train_label,test_feature,test_label):
    train_fit = fit_goodness(model,train_feature,train_label)
    test_fit =  fit_goodness(model,test_feature,test_label)
    return round((1-(test_fit-train_fit)/train_fit)*100,3)

def Linear_Regressor(train_feature,train_label,test_feature,test_label):
    # Creating a linear regression model
    lin_reg = linear_model.LinearRegression(normalize=False)
    lin_reg.fit(train_feature, train_label)
    # results of a linear model
    print('RMSE in the training dataset: ', fit_goodness(lin_reg,train_feature,train_label))
    print('RMSE in the test dataset: ',fit_goodness(lin_reg,test_feature,test_label))
    print(model_accuracy(lin_reg,train_feature,train_label,test_feature,test_label),'% of RMSE is due to the errors in training dataset')
    return lin_reg

def Ridge_Regressor(train_feature,train_label,test_feature,test_label):
    # creating a rig regression model (L2 regressor)
    alphas = np.logspace(-2,0,10)
    # NonZeroCoeff_L2: the number of non-zero coefficients for all alphas
    # Model_Coeff_L2 : the values of all the coefficients for all alphas
    Train_Error_L2 = np.zeros(shape=(len(alphas),1))
    Test_Error_L2 = np.zeros(shape=(len(alphas),1))

    Model_Coeff_L2  = np.zeros(shape=(train_feature.shape[1], len(alphas)))
    NonZeroCoeff_L2 = np.zeros(shape=(len(alphas),1))
    count = -1
    for alpha in alphas:
        count += 1
        # instantiate a logistic regression model, and fit with X and y
        model_L2 = linear_model.Ridge(alpha=alpha)
        # Fit the model
        model_L2.fit(train_feature, train_label)
        Train_Error_L2[count,0] = fit_goodness(model_L2,train_feature, train_label)
        Test_Error_L2[count,0]  = fit_goodness(model_L2,test_feature, test_label)
        Model_Coeff_L2[:,count] = model_L2.coef_
        NonZeroCoeff_L2[count,0] = train_feature.shape[1] - (model_L2.coef_ == 0).sum()
    fig = plt.figure()
    plt.scatter(alphas, Test_Error_L2, label='Test', s = 50)
    plt.scatter(alphas, Train_Error_L2, label='Train', s = 50)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Standard Mean Square Error')
    plt.legend()
    fig.set_size_inches(6, 10)
    fig.savefig('figures/Ridge_regressor_train.png', dpi=100)
    plt.show()
    print('Number of Non zero Coeff: ',NonZeroCoeff_L2)
    return model_L2

def Ridge_Regressor_CV(train_feature,train_label,test_feature,test_label):
    splitter = KFold(5, random_state=42, shuffle=True)
    alphas = np.logspace(-2,1,50)
    model_RidgeCV = linear_model.RidgeCV(alphas=alphas, cv=splitter).fit(train_feature,train_label)
    print('Number of tested alphas: ',alphas.shape)
    print('best alpha: ',round(model_RidgeCV.alpha_,3))
    print('Non zero coefficients: ',train_feature.shape[1] - (model_RidgeCV.coef_ == 0).sum())
    print('Training Standard mean sqr error: ', fit_goodness(model_RidgeCV,train_feature,train_label))
    print('Test Standard mean sqr error: ', fit_goodness(model_RidgeCV,test_feature,test_label))
    return model_RidgeCV
def Lasso_Regressor(train_feature,train_label,test_feature,test_label):
    alphas = np.logspace(-1,0,20)
    print(alphas)
    # creating a Lasso regression
    Train_Error_L1 = np.zeros(shape=(len(alphas),1))
    Test_Error_L1 = np.zeros(shape=(len(alphas),1))
    Model_Coeff_L1  = np.zeros(shape=(train_feature.shape[1], len(alphas)))
    NonZeroCoeff_L1 = np.zeros(shape=(len(alphas),1))
    count = -1
    for alpha in alphas:
        count += 1
        # instantiate a logistic regression model, and fit with X and y
        model_L1 = linear_model.Lasso(alpha=alpha, max_iter=2000)
        # Fit the model
        model_L1.fit(train_feature, train_label)
        Train_Error_L1[count,0] = fit_goodness(model_L1,train_feature, train_label)
        Test_Error_L1[count,0]  = fit_goodness(model_L1,test_feature, test_label)
        Model_Coeff_L1[:,count] = model_L1.coef_
        NonZeroCoeff_L1[count,0] = train_feature.shape[1] - (model_L1.coef_ == 0).sum()

    fig = plt.figure()
    plt.scatter(alphas, Test_Error_L1, label='Test', s = 50)
    plt.scatter(alphas, Train_Error_L1, label='Train', s = 50)
    plt.title(r'Standard Mean Square Error as a function of $\alpha$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Standard Mean Square Error')
    plt.legend()
    fig.set_size_inches(6, 10)
    fig.savefig('figures/Lasso_regressor_validation.png', dpi=100)
    plt.show()
    
    fig = plt.figure()
    plt.scatter(alphas,NonZeroCoeff_L1, label = 'non-zero coeff', s = 50)
    plt.legend()
    plt.title(r'Number of non-zero coefficients as a function of $\alpha$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Count of non-zero coefficients')
    fig.set_size_inches(6, 10)
    fig.savefig('figures/Lasso_regressor_Nonzer_validation.png', dpi=100)
    plt.show()

    print('Number of Non zero Coeff: ', NonZeroCoeff_L1)
    return model_L1

def Lasso_Regressor_CV(train_feature,train_label,test_feature,test_label):
    ## Cross validation Lasso Regression
    splitter = KFold(5, random_state=42, shuffle=True)
    alphas = np.logspace(-2, 0, 50)
    model_LassoCV = linear_model.LassoCV(alphas=alphas, cv=splitter, max_iter=10000).fit(train_feature, train_label)
    print('Number of tested alphas: ',alphas.shape)
    print('best alpha: ',round(model_LassoCV.alpha_,3))
    print('Non zero coefficients: ',train_feature.shape[1] - (model_LassoCV.coef_ == 0).sum())
    print('Training Standard mean sqr error: ', fit_goodness(model_LassoCV,train_feature,train_label))
    print('Test Standard mean sqr error: ', fit_goodness(model_LassoCV,test_feature,test_label))
    return model_LassoCV

def RandomForest_Regressor(train_feature,train_label,test_feature,test_label, estimator): 
    regr = RandomForestRegressor(n_estimators=estimator,random_state=42)
    regr.fit(train_feature,train_label)
    fi = regr.feature_importances_
    print('feature importance :',fi.shape)
    print('Training Standard mean sqr error: ', fit_goodness(regr,train_feature,train_label))
    print('Test Standard mean sqr error: ', fit_goodness(regr,test_feature,test_label))
    return regr

def RandomForest_Regressor_CV(train_feature,train_label,test_feature,test_label):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1200, num = 4)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 3)]
    # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                #'min_samples_split': min_samples_split,
                # 'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    print(random_grid)
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
    rf_random.fit(train_feature, train_label)
    print('Training Standard mean sqr error: ', fit_goodness(rf_random,train_feature,train_label))
    print('Test Standard mean sqr error: ', fit_goodness(rf_random,test_feature,test_label))
    print('best parameters for the Random Forest Model: ', rf_random.best_params_)
    return rf_random
#%% [markdown]
## Traning Regrossors
#%%
x,y = read_ChEMBL()
# %%
# Output density shape
Kernel_Density_Estimator(y)
# data splitting
# %%
X_train, X_test, y_train, y_test = Data_Splitting(x,y, 0.25)
# %%
# regressors training
m1 =    Linear_Regressor    (X_train,y_train,X_test,y_test)
# %%
m2 =    Ridge_Regressor     (X_train,y_train,X_test,y_test)
# %%
m2_CV = Ridge_Regressor_CV  (X_train,y_train,X_test,y_test)
# %%
m3 =    Lasso_Regressor     (X_train,y_train,X_test,y_test)
# %%
m3_CV = Lasso_Regressor_CV  (X_train,y_train,X_test,y_test)
# %%
m4 =    RandomForest_Regressor(X_train,y_train,X_test,y_test,100)
# %%
m4_CV = RandomForest_Regressor_CV(X_train,y_train,X_test,y_test)
# %%
#%% [markdown]
## Validation
#%%
X_Validation , Y_Validation = read_D3C()
#%%
print('Lasso Regression:: Test Standard mean sqr error for validation: ', fit_goodness(m3_CV,X_Validation,Y_Validation))
# %%
print('Ridge Regression:: Test Standard mean sqr error for validation: ', fit_goodness(m2_CV,X_Validation,Y_Validation))
# %%
print('RandomForest:: Test Standard mean sqr error for validation: ', fit_goodness(m4_CV,X_Validation,Y_Validation))

#%%
print('Linear Regression:: Test Standard mean sqr error for validation: ', fit_goodness(m1,X_Validation,Y_Validation))

# %%
# X_train_2,Y_train_2 = read_ChEMBL()
#%%
# X_test_2 , Y_test_2 = read_D3C()
#%%
# Kernel_Density_Estimator(Y_test_2)
# %%
# regressors training
# m1 =    Linear_Regressor    (X_train_2,Y_train_2,X_test_2,Y_test_2)
# %%
# m2 =    Ridge_Regressor     (X_train_2,Y_train_2,X_test_2,Y_test_2)
# %%
# m2_CV = Ridge_Regressor_CV  (X_train_2,Y_train_2,X_test_2,Y_test_2)
# %%
# m3 =    Lasso_Regressor     (X_train_2,Y_train_2,X_test_2,Y_test_2)
# %%
# m3_CV = Lasso_Regressor_CV  (X_train_2,Y_train_2,X_test_2,Y_test_2)

# %%
# m4 =    RandomForest_Regressor(X_train_2,Y_train_2,X_test_2,Y_test_2,100)
# %%
#m4_CV = RandomForest_Regressor_CV(X_train_2,Y_train_2,X_test_2,Y_test_2)

# %%

# %%
