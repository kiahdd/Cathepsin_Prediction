# %%
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit import RDConfig
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

def get_Descriptors(smiles, ind ):
    df= pd.DataFrame({'smiles': smiles}, index= ind)
    df['molecules'] = df.smiles.apply(Chem.MolFromSmiles)
    descriptors = { 'MolLogP':Descriptors.MolLogP,
            'HeavyAtomCount':Descriptors.HeavyAtomCount,
            'HeavyAtomMolWt':Descriptors.HeavyAtomMolWt, 
            'HAccept':Descriptors.NumHAcceptors,
            'Heteroatoms':Descriptors.NumHeteroatoms,
            'HDonor':Descriptors.NumHDonors,
            'MolWt':Descriptors.MolWt,
            'MolMR':Descriptors.MolMR, 
            'RotableBonds':Descriptors.NumRotatableBonds
            ,'RingCount':Descriptors.RingCount,
            'Ipc':Descriptors.Ipc
            ,'HallKierAlpha':Descriptors.HallKierAlpha
            ,'NumValenceElectrons':Descriptors.NumValenceElectrons
            ,'SaturatedRings':Descriptors.NumSaturatedRings
            ,'AliphaticRings':Descriptors.NumAliphaticRings
            ,'AromaticRings':Descriptors.NumAromaticRings
            ,'FpDensityMorgan1':Descriptors.FpDensityMorgan1, 
            'FpDensityMorgan2':Descriptors.FpDensityMorgan2, 
            'FpDensityMorgan3':Descriptors.FpDensityMorgan3, 
            'MaxAbsPartialCharge':Descriptors.MaxAbsPartialCharge, 
            'MaxPartialCharge':Descriptors.MaxPartialCharge, 
            'MinAbsPartialCharge':Descriptors.MinAbsPartialCharge, 
            'MinPartialCharge':Descriptors.MinPartialCharge, 
            'NumRadicalElectrons':Descriptors.NumRadicalElectrons, 
            'BalabanJ':Descriptors.BalabanJ, 
            'BertzCT':Descriptors.BertzCT, 
            'Kappa1':Descriptors.Kappa1, 
            'Kappa2':Descriptors.Kappa2, 
            'Kappa3':Descriptors.Kappa3, 
            'Chi0':Descriptors.Chi0, 
            'Chi1':Descriptors.Chi1, 
            'Chi0n':Descriptors.Chi0n, 
            'Chi1n':Descriptors.Chi1n, 
            'Chi2n':Descriptors.Chi2n, 
            'Chi3n':Descriptors.Chi3n, 
            'Chi4n':Descriptors.Chi4n, 
            'Chi0v':Descriptors.Chi0v, 
            'Chi1v':Descriptors.Chi1v, 
            'Chi2v':Descriptors.Chi2v, 
            'Chi3v':Descriptors.Chi3v, 
            'Chi4v':Descriptors.Chi4v, 
            'NHOHCount':Descriptors.NHOHCount, 
            'NOCount':Descriptors.NOCount, 
            'FractionCSP3':Descriptors.FractionCSP3, 
            'LabuteASA':Descriptors.LabuteASA, 
            'PEOE_VSA1':Descriptors.PEOE_VSA1,
            'PEOE_VSA2':Descriptors.PEOE_VSA2, 
            'PEOE_VSA3':Descriptors.PEOE_VSA3, 
            'PEOE_VSA4':Descriptors.PEOE_VSA4, 
            'PEOE_VSA5':Descriptors.PEOE_VSA5, 
            'PEOE_VSA6':Descriptors.PEOE_VSA6, 
            'PEOE_VSA7':Descriptors.PEOE_VSA7,
            'PEOE_VSA8':Descriptors.PEOE_VSA8, 
            'PEOE_VSA9':Descriptors.PEOE_VSA9, 
            'PEOE_VSA10':Descriptors.PEOE_VSA10, 
            'PEOE_VSA11':Descriptors.PEOE_VSA11, 
            'PEOE_VSA12':Descriptors.PEOE_VSA12, 
            'PEOE_VSA13':Descriptors.PEOE_VSA13, 
            'PEOE_VSA14':Descriptors.PEOE_VSA14, 
            'SMR_VSA1':Descriptors.SMR_VSA1,
            'SMR_VSA2':Descriptors.SMR_VSA2, 
            'SMR_VSA3':Descriptors.SMR_VSA3, 
            'SMR_VSA4':Descriptors.SMR_VSA4, 
            'SMR_VSA5':Descriptors.SMR_VSA5, 
            'SMR_VSA6':Descriptors.SMR_VSA6, 
            'SMR_VSA7':Descriptors.SMR_VSA7, 
            'SMR_VSA8':Descriptors.SMR_VSA8, 
            'SMR_VSA9':Descriptors.SMR_VSA9, 
            'SMR_VSA10':Descriptors.SMR_VSA10, 
            'SlogP_VSA1':Descriptors.SlogP_VSA1, 
            'SlogP_VSA2':Descriptors.SlogP_VSA2, 
            'SlogP_VSA3':Descriptors.SlogP_VSA3, 
            'SlogP_VSA4':Descriptors.SlogP_VSA4, 
            'SlogP_VSA5':Descriptors.SlogP_VSA5, 
            'SlogP_VSA6':Descriptors.SlogP_VSA6, 
            'SlogP_VSA7':Descriptors.SlogP_VSA7, 
            'SlogP_VSA8':Descriptors.SlogP_VSA8, 
            'SlogP_VSA9':Descriptors.SlogP_VSA9, 
            'SlogP_VSA10':Descriptors.SlogP_VSA10, 
            'SlogP_VSA11':Descriptors.SlogP_VSA11, 
            'SlogP_VSA12':Descriptors.SlogP_VSA12, 
            'EState_VSA1':Descriptors.EState_VSA1, 
            'EState_VSA2':Descriptors.EState_VSA2, 
            'EState_VSA3':Descriptors.EState_VSA3, 
            'EState_VSA4':Descriptors.EState_VSA4, 
            'EState_VSA5':Descriptors.EState_VSA5, 
            'EState_VSA6':Descriptors.EState_VSA6, 
            'EState_VSA7':Descriptors.EState_VSA7, 
            'EState_VSA8':Descriptors.EState_VSA8, 
            'EState_VSA9':Descriptors.EState_VSA9, 
            'EState_VSA10':Descriptors.EState_VSA10, 
            'EState_VSA11':Descriptors.EState_VSA11, 
            'VSA_EState1':Descriptors.VSA_EState1, 
            'VSA_EState2':Descriptors.VSA_EState2, 
            'VSA_EState3':Descriptors.VSA_EState3, 
            'VSA_EState4':Descriptors.VSA_EState4, 
            'VSA_EState5':Descriptors.VSA_EState5, 
            'VSA_EState6':Descriptors.VSA_EState6, 
            'VSA_EState7':Descriptors.VSA_EState7, 
            'VSA_EState8':Descriptors.VSA_EState8, 
            'VSA_EState9':Descriptors.VSA_EState9, 
            'VSA_EState10':Descriptors.VSA_EState10}
    for key, val in descriptors.items():
        df[key] = df.molecules.apply(val)

    return df

# df_d3c = pd.read_csv('data/CatS_score.csv')
# df_feats_d3c = get_Descriptors(df_d3c['SMILES'].values, 
# ind= df_d3c['Cmpd_ID'].values)

# print(df_feats_d3c.head())

# %%
