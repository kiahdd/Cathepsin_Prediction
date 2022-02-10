# %%
import numpy as np
import pandas as pd
from pandas import DataFrame
#importing rdkit
from rdkit import Chem
from rdkit import DataStructs

# %%
## analyzing features
compounds = pd.read_csv('data/cleaned_CatS_CHEMBL.csv', index_col='ID')
print(compounds.shape)
print(compounds.info())
# %%
compounds['molecules'] = compounds.smiles.apply(Chem.MolFromSmiles)
compounds['molecules']
# %%
# Molecular Fingerprints
compounds['fingerprints'] = compounds.molecules.apply(Chem.RDKFingerprint)
compounds['fingerprints'] 

# %%
## test run similarities
DataStructs.FingerprintSimilarity(compounds['fingerprints'][0],compounds['fingerprints'][1])

# %%
## Filling the matrix
counter = 0
for comp1 in compounds.fingerprints.iteritems():
    for comp2 in compounds.fingerprints.iteritems():
        score = DataStructs.FingerprintSimilarity(comp1[1],comp2[1])
        matrix_similarity.loc[comp1[0],comp2[0]] = score
    print(counter)
    counter += 1

# %%
matrix_similarity.head(20)
# %%
matrix_similarity.to_csv ('data/matrix_similarities.csv', index = True, header=True)

# %%