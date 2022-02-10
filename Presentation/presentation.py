#%%
import chart_studio.plotly as py
import chart_studio.presentation_objs as pres
import json

filename = 'CHE1147_Project_Presentation'

with open('pres_1.json', 'r') as f:
    my_pres = json.load(f)

# my_pres = pres.Presentation(markdown_string)

pres_url_1 = py.presentation_ops.upload(my_pres, filename)

print(pres_url_1)

# %%
