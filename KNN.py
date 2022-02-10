#%%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

# %%
data = pd.read_csv('data/Unique_features.csv', index_col = 'ID')
data.head()

pca_data = pd.read_csv('data/pca_scores.csv')
pca_data.head()

D3_data = pd.read_csv('data/D3C_clean_features.csv', index_col= 'Cmpd_ID')
# %%
X = data.loc[:, 'MolLogP':'VSA_EState10']
y = data.loc[:, 'pchembl']

X_test_real = D3_data.loc[:, 'MolLogP':'VSA_EState10']
y_test_real = D3_data.loc[:, 'pIC50']

X_test_real
#%%
#Convert output to boolean

for i in range(y.shape[0]):
    if y[i] >=7:
       y[i] = int(1)
        #print('YES!!!!')
    else:
       y[i] = int(0)

for i in range(y_test_real.shape[0]):
    if y_test_real[i] >=7:
       y_test_real[i] = int(1)
        #print('YES!!!!')
    else:
       y_test_real[i] = int(0)
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# %%

#Scaling the features using standard scaler ((x-mean)/std_dev)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_test_real = scaler.transform(X_test_real)

# %%
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# %%
plt.figure(figsize=(12, 6))
plt.plot(range(1, 100), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# %%
classifier = KNeighborsClassifier(n_neighbors= 5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test_real)

# %%
print(confusion_matrix(y_test_real, y_pred))
print(classification_report(y_test_real, y_pred))


# %%
title = 'Normalized confusion matrix using KNN'
disp = plot_confusion_matrix(classifier, X_test_real, y_test_real,
                                display_labels=['Weak Inhibitor', 'Strong Inhibitor'],
                                cmap=plt.cm.Blues,
                                normalize='true')
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)

plt.show()

# %%
