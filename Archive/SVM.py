#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing as prep
from sklearn.metrics import plot_confusion_matrix
# %%
data = pd.read_csv('data/features_out_cat.csv')
data.head()

# %%
X = data.loc[:, 'MolLogP': 'VSA_EState10']
Y = data['Binary_value']

# Data normalization
MinMax_scaler = prep.MinMaxScaler()
X[X.columns] = MinMax_scaler.fit_transform(X[X.columns])

Standard_scaler = prep.StandardScaler()
X['Ipc'] = Standard_scaler.fit_transform(data['Ipc'].values.reshape(-1, 1))

desc = X.describe()
desc.sort_values(by=desc.index[1], axis=1,  ascending= False,  inplace=True)

desc.head()

# %%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

print('Ratio number of instances in the Test/ Train: {}'.format(len(Y_test)/len(Y_train)))

print('Ratio of 1 in Train set: {}'.format(Y_train.sum()/len(Y_train)))

print('Ratio of 1 in Test set: {}'.format(Y_test.sum()/len(Y_test)))

# %%
svcClassifier = SVC(kernel = 'linear', C = 1, gamma = 'auto', random_state= 42)
svcClassifier.fit(X_train, Y_train)

Y_pred = svcClassifier.predict(X_test)
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


title = 'Normalized confusion matrix using SVC'
disp = plot_confusion_matrix(svcClassifier, X_test, Y_test,
                                display_labels=['Weak Inhibitor', 'Strong Inhibitor'],
                                cmap=plt.cm.Blues,
                                normalize='true')
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)

plt.show()
plt.savefig('figures/SVC_Cmatrix.png')
# %%
scores = cross_val_score(svcClassifier, X,Y, cv = 5)
scores

# %%
