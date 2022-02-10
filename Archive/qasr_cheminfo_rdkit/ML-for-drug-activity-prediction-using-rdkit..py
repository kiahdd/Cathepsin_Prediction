# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'D:\Code\Machine Learning\ML_projects\qasr_cheminfo_rdkit'))
	print(os.getcwd())
except:
	pass
# %%
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import emoji
get_ipython().run_line_magic('matplotlib', 'inline')

#importing rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import RDConfig
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from utility import FeatureGenerator
from rdkit.Chem import PandasTools as PandasTools
from rdkit import DataStructs
from rdkit.Chem.Subshape import SubshapeBuilder,SubshapeAligner,SubshapeObjects

#importing sklearn 
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# %% [markdown]
# **1. Scientific/Business Understanding**
# 
# Being a scientist and eager to learn new method in my researech field, exploring chemoinformatics approach in drug discovery is quite interesting. For this project I am using the experimental *log P* data released in the paper: “Large, chemically diverse dataset of *log P* measurements for benchmarking studies” by Martel et al: https://doi.org/10.1016/j.ejps.2012.10.019. As this is a preliminary study, we are interested in finding which featurization methods work best for predicting *log P*. The octanol-water partition coefficient (*log P*), is one of the most important properties for determining a compound’s suitability as a drug. This coefficient strongly affects how easily the drug can reach its intended target in the body, how strong an effect it will have once it reaches its target, and how long it will remain in the body in an active form, hence used in decision-making by medicinal chemists in pre-clinical drug discovery. The raw data is downloaded from https://ochem.eu/article/27772.
# 
# I will focus on the three question below:
# 
# Q1: What are are options for *log P* prediction are based on physical descriptors, such as atom type counts, or polar surface area, or on topological descriptors?
# 
# Q2: What's the *log P* distribution like? How is it related to drug likeliness?
# 
# Q3: How is molecular descriptor have effect on *log P*? Can we predict *log P* based on molecular descriptor set?
# %% [markdown]
# **2. Data understanding**

# %%
#load data
df = pd.read_csv("data.csv")
df.head()


# %%
#number of rows and column
print(f'No. of rows {df.shape[0]}, no of columns {df.shape[1]}')


# %%
#summary statistics
df.describe()   


# %%
#missing values
df.isnull().sum(axis=0)


# %%

# data format for each column
df.info()


# %%
df.columns

# %% [markdown]
# **3. Prepare Data**
# 
# There are some necessary stpes to apply before continue exploring the dataset:
# 
# Drop unused columns
# 
# Convert string values to number
# 

# %%
from copy import deepcopy
data = df.copy(deep=True)


# %%
# Drop unused columns
columns_to_drop = ['CASRN', 'MOLECULEID','ARTICLEID', 'PUBMEDID', 'PAGE', 'TABLE', 'UNIT {logPow}', 
                   'logPow {measured, converted}', 'UNIT {logPow}.1','UNIT {pH}', 'NAME', 'NAME.1']

data.drop(columns_to_drop, axis=1, inplace=True)

#renaming logP column
data.rename(columns={'logPow {measured}': 'logPexp', 'EXTERNALID': 'ZINC_ID'}, inplace=True)


# %%
#rearraging columns sequence
data = data[['N', 'ZINC_ID', 'SMILES', 'logPexp', 'pH']]


# %%
data.info()


# %%
data.head()


# %%
#cleaning pH column as some of the values has '=' 
data['pH'] = pd.to_numeric(data['pH'].map(lambda x: x.lstrip('=').rstrip('=')))


# %%
data.head()


# %%
data.describe()

# %% [markdown]
# #### *log P* distribution plot

# %%
sns.distplot(data['logPexp'])

# %% [markdown]
# #### pH distribution plot

# %%
sns.distplot(data['pH'])

# %% [markdown]
# We can see that there is no missing values. We drop few unnecessary columns and cleaned the pH column values and converted into interger values
# %% [markdown]
# **4. Answer Questions base on dataset**
# 
# Q1: Visualize molecules and check how simillar they are?

# %%
#Convert SMILES to 2D molecules:
molecules = data.SMILES.apply(Chem.MolFromSmiles)

# %% [markdown]
# Visualizing 20 molecules in 5*4 grid

# %%
#Display some of the moecules
Draw.MolsToGridImage(molecules[:20],molsPerRow=4)


# %%
## Similarity Calculations

# We calculate Morgan fingerprints 
mols_fps=[AllChem.GetMorganFingerprintAsBitVect(x,2) for x in molecules]
print(f'Tanimoto Similarity Coefficient of molecule_1 vs molecule_1: {DataStructs.TanimotoSimilarity(mols_fps[0],mols_fps[0])}')
print(f'Tanimoto Similarity Coefficient of molecule_1 vs molecule_2: {DataStructs.TanimotoSimilarity(mols_fps[0],mols_fps[1])}')

# Calculate similarities to reference molecule
sim_ref = DataStructs.BulkTanimotoSimilarity(mols_fps[0],mols_fps)


# Calculate pairwise similarities
def pairwise_sim(mols):
    pairwise=[]
    for i in mols:
        sim = DataStructs.BulkTanimotoSimilarity(i,mols)
        pairwise.append(sim)
    return pairwise

PW = pairwise_sim(mols_fps)
figure,(plt1,plt2) = plt.subplots(1,2)
figure.set_size_inches(15,5)
plt1.hist(sim_ref)
plt1.set_title('Similarity to reference molecule')
plt2.hist(PW)
plt2.set_title('Pairwise similarity all molecules')

# %% [markdown]
# We have generated a connectivity fingerprint using Morgan Fingerprint and based on this fingerprint calculated the similarity measure. From the result and plot from sililarity and pairwise similarity it seems that most of the molecules are quite different from the reference molecule (molecule_0). This is a good indication for having diverse set of molecule covering large chemical space. 
# %% [markdown]
# **3. Prepare Data (feature engineering)**
# 
# Next, we use RDKit to calculate some of molecular descriptors 

# %%
data.loc[:, 'MolLogP'] = molecules.apply(Descriptors.MolLogP)
data.loc[:, 'HeavyAtomCount'] = molecules.apply(Descriptors.HeavyAtomCount)
data.loc[:, 'HAccept'] = molecules.apply(Descriptors.NumHAcceptors)
data.loc[:, 'Heteroatoms'] = molecules.apply(Descriptors.NumHeteroatoms)
data.loc[:, 'HDonor'] = molecules.apply(Descriptors.NumHDonors)
data.loc[:, 'MolWt'] = molecules.apply(Descriptors.MolWt)
data.loc[:, 'RotableBonds'] = molecules.apply(Descriptors.NumRotatableBonds)
data.loc[:, 'RingCount'] = molecules.apply(Descriptors.RingCount)
data.loc[:, 'Ipc'] = molecules.apply(Descriptors.Ipc)
data.loc[:, 'HallKierAlpha'] = molecules.apply(Descriptors.HallKierAlpha)
data.loc[:, 'NumValenceElectrons'] = molecules.apply(Descriptors.NumValenceElectrons)
data.loc[:, 'SaturatedRings'] = molecules.apply(Descriptors.NumSaturatedRings)
data.loc[:, 'AliphaticRings'] = molecules.apply(Descriptors.NumAliphaticRings)
data.loc[:, 'AromaticRings'] = molecules.apply(Descriptors.NumAromaticRings)


# %%
data.head()

# %% [markdown]
# As a baseline, we calculate the performance of RDKit’s calculated MolLogP vs the experimental log P.

# %%
r2 = r2_score(data.logPexp, data.MolLogP)
mse = mean_squared_error(data.logPexp, data.MolLogP)
plt.scatter(data.logPexp, data.MolLogP,
            label = "MSE: {:.2f}\nR^2: {:.2f}".format(mse, r2))
plt.legend()
plt.show()

# %% [markdown]
# As we can see above, RDKit’s *log P* predictions have a relatively high mean square error, and a weak $R^2$ of determination for this dataset. RDKit’s MolLogP implementation is based on atomic contributions. Hence, we will first try to train our own simple *log P* model using the RDKit physical descriptors that we generated above.
# %% [markdown]
# # Simple descriptor models traning
# 
# Following descriptor we will use to train our simple model

# %%
X = data.iloc[:, 6:]
y = data.logPexp
X.head()

# %% [markdown]
# For the regression, we will use a Random Forest with the default parameters from scikit-learn, and set aside one 20\% of the data for testing.

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Default models
models = {"rf": RandomForestRegressor(n_estimators=100, random_state=42)}

scores = {}
for m in models:
    models[m].fit(X_train, y_train)
    scores[m + "_train"] = models[m].score(X_train, y_train, )
    y_pred = models[m].predict(X_test)
    scores[m + "_test"] = r2_score(y_test, y_pred)
    scores[m + "_mse_test"] = mean_squared_error(y_test, y_pred)


# %%
scores = pd.Series(scores).T
scores


# %%
r_2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
plt.scatter(y_test, y_pred, label = "MSE: {:.2f}\nR^2: {:.2f}".format(mse, r_2))
plt.legend()
plt.show()

# %% [markdown]
# As we can see, using these simple descriptors coupled with scikit-learn’s default random forest gets us a higher R2 and MSE performance than the RDKit *log P* predictor. It would be interesting to see how much we can improve the performance by tuning the random forest parameters.

# %%
param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}


# %%
estimator = RandomForestRegressor()


# %%
grid = GridSearchCV(estimator, param_grid, n_jobs=-1, cv=5)


# %%
grid.fit(X_train, y_train)


# %%
grid.best_estimator_


# %%
grid.best_params_


# %%
models = {"rf": RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=200,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)}

scores = {}
for m in models:
    models[m].fit(X_train, y_train)
    scores[m + "_train"] = models[m].score(X_train, y_train, )
    y_pred = models[m].predict(X_test)
    scores[m + "_test"] = r2_score(y_test, y_pred)
    scores[m + "_mse_test"] = mean_squared_error(y_test, y_pred)


# %%
scores = pd.Series(scores).T
scores


# %%
r_2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
plt.scatter(y_test, y_pred, label = "MSE: {:.2f}\nR^2: {:.2f}".format(mse, r_2))
plt.legend()
plt.show()

# %% [markdown]
# Hyperparameter tuning which essentially relies more on experimental results than theory, and thus the best method to determine the optimal settings is to try many different combinations evaluate the performance of each model. In our case its seems to improve the estimates a bit but not very significant. 
# %% [markdown]
# # Continue Preparing Data for Q3
# 
# ## Calculating some more advanced moleculating fingerprint ##
# %% [markdown]
# We will test Morgan fingerprints (ECFP4 and ECFP6), RDKFingerprints, and topological pharmacophore fingerprints (TPAPF and TPATF), the scripts for which are available from [MayaChemTools](http://www.mayachemtools.org/).
# %% [markdown]
# Just note here that I am running this notebook on a super computing [PL-Grid Infrastructure](http://www.plgrid.pl/en) where each computing node has 24 cpu and 128 GB of memory which is quite impressive.To utlize the max cpu utilization lets create a function for parallelizing a DataFrame’s apply() function. This makes TPATF and TPAPF fingerprint calculation much faster. 

# %%
import multiprocessing
from joblib import Parallel, delayed

def applyParallel(df, func):
    """This function splits a pandas Series into n chunks,
    corresponding to the number of available CPUs. Then it
    applies a given function to the dataframe chunks, and 
    finally, returns their concatenated output."""
    n_jobs=multiprocessing.cpu_count()
    groups =  np.array_split(df, n_jobs)
    results = Parallel(n_jobs)(delayed(lambda g: g.apply(func))(group) for group in groups)
    return pd.concat(results)

# %% [markdown]
# Calculate fingerprints:

# %%
fps = {"ECFP4": molecules.apply(lambda m: AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)),
       "ECFP6": molecules.apply(lambda m: AllChem.GetMorganFingerprintAsBitVect(m, radius=3, nBits=2048)),
       "RDKFP": molecules.apply(lambda m: AllChem.RDKFingerprint(m, fpSize=2048)),
       "TPATF": applyParallel(data.SMILES, lambda m: FeatureGenerator(m).toTPATF()),
       "TPAPF": applyParallel(data.SMILES, lambda m: FeatureGenerator(m).toTPAPF())}


# %%
fps.keys()


# %%


# %% [markdown]
# # 5. Train model and Measure Performance
# 
# Finally, here we apply three different types of regression models to estimate the performance of these calculated different fingerprints.

# %%
# Default models
models = {"rf": RandomForestRegressor(n_estimators=100, random_state=42),
          "nnet": MLPRegressor(random_state=42),
          "svr": SVR(gamma='auto')}

scores = {}

for f in fps:
    scores[f] = {}
    # Convert fps to 2D numpy array
    X = np.array(fps[f].tolist())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    for m in models:
        models[m].fit(X_train, y_train)
        y_pred = models[m].predict(X_test)
        scores[f][m + "_r2_test"] = r2_score(y_test, y_pred)
        scores[f][m + "_mse_test"] = mean_squared_error(y_test, y_pred)
        


# %%
scores_df = pd.DataFrame(scores).T
scores_df

# %% [markdown]
# The Topological Pharmacophore Atomic Triplets Fingerprint (TPATF) performed the best - even outperforming the simple descriptor model. The default random forest had the best performance out of all the regression methods, although it is subject to change after hyperparameter tuning. 
# 
# The above exercise shows some aspect of machine leaning in chemistry and classically called as Quantitative Structure Activity prediction (QSAR) and is in practice for several decades. Actually there is lot use cases of machine leaning in the field of physical chemistry such as designing new materials, new energy source, etc. 

# %%



# %%


