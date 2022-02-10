# This script is to extract information from the CHEMBL Database. In specific the available bioactive data for the target Cathepsin S in Humans. 

#%%

from chembl_webresource_client.new_client import new_client
import pandas as pd
from pandas import Series,DataFrame 
import numpy as np 
import matplotlib.pyplot as plt

target_id = 'CHEMBL2954' # CHEMBL ID for Cathepsin S in Humans 
organism = 'Homo Sapiens'

# available_resources = [resource for resource in dir(new_client) if not resource.startswith('_')]
# print (available_resources)   

activity = new_client.activity
res = activity.filter(target_chembl_id=target_id, standard_type__iregex='(IC50|Ki|EC50|Kd)')
fdata = dict()
for key, values in res[1].items(): 
    fdata[key] = []
for i in res:
    for k , v in i.items():
        fdata[k].append(v)

df_cats = DataFrame.from_dict(fdata)
df_cats.to_csv('CatS_CHEMBL.csv')
print(df_cats.head())



# %%
for key, values in res[2].items(): 
    print('{} /// {}'.format(key, values))


# %%
