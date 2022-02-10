#%%

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#importing rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors3D
from rdkit import RDConfig
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
#from utility import FeatureGenerator
from rdkit.Chem import PandasTools as PandasTools
from rdkit.Chem.Subshape import SubshapeBuilder,SubshapeAligner,SubshapeObjects

#importing sklearn 
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# %%

df = pd.read_csv('data\cleaned_CatS_CHEMBL.csv')
smiles_list = df['smiles'].values
ic50_list =df['IC50'].values
# print(smiles_list[0])
# print(ic50_list[0])

# %%

molecules = df.smiles.apply(Chem.MolFromSmiles)
#Display some of the molecules
Draw.MolsToGridImage(molecules[:20],molsPerRow=4)

# %%

# TODO: Add more relevant features from the available features from RDkit.

# use RDKit to calculate some of molecular descriptors 
df.loc[:, 'MolLogP'] = molecules.apply(Descriptors.MolLogP)
df.loc[:, 'HeavyAtomCount'] = molecules.apply(Descriptors.HeavyAtomCount)
df.loc[:, 'HeavyAtomMolWt'] = molecules.apply(Descriptors.HeavyAtomMolWt) 
df.loc[:, 'HAccept'] = molecules.apply(Descriptors.NumHAcceptors)
df.loc[:, 'Heteroatoms'] = molecules.apply(Descriptors.NumHeteroatoms)
df.loc[:, 'HDonor'] = molecules.apply(Descriptors.NumHDonors)
df.loc[:, 'MolWt'] = molecules.apply(Descriptors.MolWt)
df.loc[:, 'MolMR'] = molecules.apply(Descriptors.MolMR)  
df.loc[:, 'RotableBonds'] = molecules.apply(Descriptors.NumRotatableBonds)
df.loc[:, 'RingCount'] = molecules.apply(Descriptors.RingCount)
df.loc[:, 'Ipc'] = molecules.apply(Descriptors.Ipc)
df.loc[:, 'HallKierAlpha'] = molecules.apply(Descriptors.HallKierAlpha)
df.loc[:, 'NumValenceElectrons'] = molecules.apply(Descriptors.NumValenceElectrons)
df.loc[:, 'SaturatedRings'] = molecules.apply(Descriptors.NumSaturatedRings)
df.loc[:, 'AliphaticRings'] = molecules.apply(Descriptors.NumAliphaticRings)
df.loc[:, 'AromaticRings'] = molecules.apply(Descriptors.NumAromaticRings)
df.loc[:, 'FpDensityMorgan1'] = molecules.apply(Descriptors.FpDensityMorgan1) 
df.loc[:, 'FpDensityMorgan2'] = molecules.apply(Descriptors.FpDensityMorgan2) 
df.loc[:, 'FpDensityMorgan3'] = molecules.apply(Descriptors.FpDensityMorgan3) 
df.loc[:, 'MaxAbsPartialCharge'] = molecules.apply(Descriptors.MaxAbsPartialCharge) 
df.loc[:, 'MaxPartialCharge'] = molecules.apply(Descriptors.MaxPartialCharge) 
df.loc[:, 'MinAbsPartialCharge'] = molecules.apply(Descriptors.MinAbsPartialCharge) 
df.loc[:, 'MinPartialCharge'] = molecules.apply(Descriptors.MinPartialCharge) 
df.loc[:, 'NumRadicalElectrons'] = molecules.apply(Descriptors.NumRadicalElectrons) 
# .####################
# df.loc[:, 'Gasteiger/Marsili Partial Charges'] = molecules.apply(Descriptors.GasteigerCharges)    

df.loc[:, 'BalabanJ'] = molecules.apply(Descriptors.BalabanJ)  
df.loc[:, 'BertzCT'] = molecules.apply(Descriptors.BertzCT)  
df.loc[:, 'Kappa1'] = molecules.apply(Descriptors.Kappa1)  
df.loc[:, 'Kappa2'] = molecules.apply(Descriptors.Kappa2)  
df.loc[:, 'Kappa3'] = molecules.apply(Descriptors.Kappa3)  
df.loc[:, 'Chi0'] = molecules.apply(Descriptors.Chi0)  
df.loc[:, 'Chi1'] = molecules.apply(Descriptors.Chi1) 
df.loc[:, 'Chi0n'] = molecules.apply(Descriptors.Chi0n)  
df.loc[:, 'Chi1n'] = molecules.apply(Descriptors.Chi1n) 
df.loc[:, 'Chi2n'] = molecules.apply(Descriptors.Chi2n) 
df.loc[:, 'Chi3n'] = molecules.apply(Descriptors.Chi3n) 
df.loc[:, 'Chi4n'] = molecules.apply(Descriptors.Chi4n) 
df.loc[:, 'Chi0v'] = molecules.apply(Descriptors.Chi0v)  
df.loc[:, 'Chi1v'] = molecules.apply(Descriptors.Chi1v)  
df.loc[:, 'Chi2v'] = molecules.apply(Descriptors.Chi2v)  
df.loc[:, 'Chi3v'] = molecules.apply(Descriptors.Chi3v)  
df.loc[:, 'Chi4v'] = molecules.apply(Descriptors.Chi4v)  
df.loc[:, 'NHOHCount'] = molecules.apply(Descriptors.NHOHCount)  
df.loc[:, 'NOCount'] = molecules.apply(Descriptors.NOCount)  
df.loc[:, 'FractionCSP3'] = molecules.apply(Descriptors.FractionCSP3)  
# df.loc[:, 'NumSpiroAtoms'] = molecules.apply(Descriptors.NumSpiroAtoms)  
# df.loc[:, 'NumBridgeheadAtoms'] = molecules.apply(Descriptors.NumBridgeheadAtoms)  
df.loc[:, 'LabuteASA'] = molecules.apply(Descriptors.LabuteASA)  
df.loc[:, 'PEOE_VSA1'] = molecules.apply(Descriptors.PEOE_VSA1)
df.loc[:, 'PEOE_VSA2'] = molecules.apply(Descriptors.PEOE_VSA2) 
df.loc[:, 'PEOE_VSA3'] = molecules.apply(Descriptors.PEOE_VSA3)  
df.loc[:, 'PEOE_VSA4'] = molecules.apply(Descriptors.PEOE_VSA4) 
df.loc[:, 'PEOE_VSA5'] = molecules.apply(Descriptors.PEOE_VSA5) 
df.loc[:, 'PEOE_VSA6'] = molecules.apply(Descriptors.PEOE_VSA6) 
df.loc[:, 'PEOE_VSA7'] = molecules.apply(Descriptors.PEOE_VSA7)
df.loc[:, 'PEOE_VSA8'] = molecules.apply(Descriptors.PEOE_VSA8) 
df.loc[:, 'PEOE_VSA9'] = molecules.apply(Descriptors.PEOE_VSA9) 
df.loc[:, 'PEOE_VSA10'] = molecules.apply(Descriptors.PEOE_VSA10) 
df.loc[:, 'PEOE_VSA11'] = molecules.apply(Descriptors.PEOE_VSA11) 
df.loc[:, 'PEOE_VSA12'] = molecules.apply(Descriptors.PEOE_VSA12) 
df.loc[:, 'PEOE_VSA13'] = molecules.apply(Descriptors.PEOE_VSA13) 
df.loc[:, 'PEOE_VSA14'] = molecules.apply(Descriptors.PEOE_VSA14)   
df.loc[:, 'SMR_VSA1'] = molecules.apply(Descriptors.SMR_VSA1)
df.loc[:, 'SMR_VSA2'] = molecules.apply(Descriptors.SMR_VSA2) 
df.loc[:, 'SMR_VSA3'] = molecules.apply(Descriptors.SMR_VSA3) 
df.loc[:, 'SMR_VSA4'] = molecules.apply(Descriptors.SMR_VSA4) 
df.loc[:, 'SMR_VSA5'] = molecules.apply(Descriptors.SMR_VSA5) 
df.loc[:, 'SMR_VSA6'] = molecules.apply(Descriptors.SMR_VSA6) 
df.loc[:, 'SMR_VSA7'] = molecules.apply(Descriptors.SMR_VSA7) 
df.loc[:, 'SMR_VSA8'] = molecules.apply(Descriptors.SMR_VSA8) 
df.loc[:, 'SMR_VSA9'] = molecules.apply(Descriptors.SMR_VSA9) 
df.loc[:, 'SMR_VSA10'] = molecules.apply(Descriptors.SMR_VSA10) 
df.loc[:, 'SlogP_VSA1'] = molecules.apply(Descriptors.SlogP_VSA1) 
df.loc[:, 'SlogP_VSA2'] = molecules.apply(Descriptors.SlogP_VSA2) 
df.loc[:, 'SlogP_VSA3'] = molecules.apply(Descriptors.SlogP_VSA3) 
df.loc[:, 'SlogP_VSA4'] = molecules.apply(Descriptors.SlogP_VSA4) 
df.loc[:, 'SlogP_VSA5'] = molecules.apply(Descriptors.SlogP_VSA5) 
df.loc[:, 'SlogP_VSA6'] = molecules.apply(Descriptors.SlogP_VSA6) 
df.loc[:, 'SlogP_VSA7'] = molecules.apply(Descriptors.SlogP_VSA7) 
df.loc[:, 'SlogP_VSA8'] = molecules.apply(Descriptors.SlogP_VSA8) 
df.loc[:, 'SlogP_VSA9'] = molecules.apply(Descriptors.SlogP_VSA9) 
df.loc[:, 'SlogP_VSA10'] = molecules.apply(Descriptors.SlogP_VSA10) 
df.loc[:, 'SlogP_VSA11'] = molecules.apply(Descriptors.SlogP_VSA11) 
df.loc[:, 'SlogP_VSA12'] = molecules.apply(Descriptors.SlogP_VSA12) 
df.loc[:, 'EState_VSA1'] = molecules.apply(Descriptors.EState_VSA1) 
df.loc[:, 'EState_VSA2'] = molecules.apply(Descriptors.EState_VSA2) 
df.loc[:, 'EState_VSA3'] = molecules.apply(Descriptors.EState_VSA3) 
df.loc[:, 'EState_VSA4'] = molecules.apply(Descriptors.EState_VSA4) 
df.loc[:, 'EState_VSA5'] = molecules.apply(Descriptors.EState_VSA5) 
df.loc[:, 'EState_VSA6'] = molecules.apply(Descriptors.EState_VSA6) 
df.loc[:, 'EState_VSA7'] = molecules.apply(Descriptors.EState_VSA7) 
df.loc[:, 'EState_VSA8'] = molecules.apply(Descriptors.EState_VSA8) 
df.loc[:, 'EState_VSA9'] = molecules.apply(Descriptors.EState_VSA9) 
df.loc[:, 'EState_VSA10'] = molecules.apply(Descriptors.EState_VSA10) 
df.loc[:, 'EState_VSA11'] = molecules.apply(Descriptors.EState_VSA11) 
df.loc[:, 'VSA_EState1'] = molecules.apply(Descriptors.VSA_EState1) 
df.loc[:, 'VSA_EState2'] = molecules.apply(Descriptors.VSA_EState2) 
df.loc[:, 'VSA_EState3'] = molecules.apply(Descriptors.VSA_EState3) 
df.loc[:, 'VSA_EState4'] = molecules.apply(Descriptors.VSA_EState4) 
df.loc[:, 'VSA_EState5'] = molecules.apply(Descriptors.VSA_EState5) 
df.loc[:, 'VSA_EState6'] = molecules.apply(Descriptors.VSA_EState6) 
df.loc[:, 'VSA_EState7'] = molecules.apply(Descriptors.VSA_EState7) 
df.loc[:, 'VSA_EState8'] = molecules.apply(Descriptors.VSA_EState8) 
df.loc[:, 'VSA_EState9'] = molecules.apply(Descriptors.VSA_EState9) 
df.loc[:, 'VSA_EState10'] = molecules.apply(Descriptors.VSA_EState10) 

print(df.head())

#%%

# df.loc[:, 'PMI1'] = molecules.apply(Descriptors3D.PMI1)  
# df.loc[:, 'PMI2'] = molecules.apply(Descriptors3D.PMI2) 
# df.loc[:, 'PMI3'] = molecules.apply(Descriptors3D.PMI3) 
# df.loc[:, 'NPR1'] = molecules.apply(Descriptors3D.NPR1)  
# df.loc[:, 'NPR2'] = molecules.apply(Descriptors3D.NPR2)  
# df.loc[:, 'Radius of gyration'] = molecules.apply(Descriptors3D.RadiusOfGyration)  
# df.loc[:, 'Asphericity'] = molecules.apply(Descriptors3D.Asphericity)  
# df.loc[:, 'Eccentricity'] = molecules.apply(Descriptors3D.Eccentricity)  
# df.loc[:, 'InertialShapeFactor'] = molecules.apply(Descriptors3D.InertialShapeFactor)  
# df.loc[:, 'SpherocityIndex'] = molecules.apply(Descriptors3D.SpherocityIndex)  

print(df.head())

# %%

features = df.iloc[:, 6:]
target_var = df.IC50
print(features.head())
df.to_csv('data/data_features_1.csv')
# %%
