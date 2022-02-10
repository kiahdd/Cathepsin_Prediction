#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing as prep
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# %%
df_out = pd.read_csv('data/features_out_cat.csv', index_col= 'ID')
df_out.head()
# %%
df_pca = pd.read_csv('data/pca_scores.csv', index_col= 'ID')
df_pca.head()

D3_pca_data = pd.read_csv('data/D3C_pca_scores.csv', index_col = 'Cmpd_ID')
D3_data.head()

D3_data = pd.read_csv('data/D3C_clean_features.csv', index_col= 'Cmpd_ID')

# %%
X = df_pca
y = df_out['Binary_value']

X_test_real = D3_pca_data
y_test_real = D3_data.iloc[:, -1]

for i in range(y_test_real.shape[0]):
    if y_test_real[i] >=7:
       y_test_real[i] = 1
        #print('YES!!!!')
    else:
       y_test_real[i] = 0

#%%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

print('Ratio number of instances in the Test/ Train: {}'.format(len(y_test)/len(y_train)))

print('Ratio of 1 in Train set: {}'.format(y_train.sum()/len(y_train)))

print('Ratio of 1 in Test set: {}'.format(y_test.sum()/len(y_test)))

#%%
#Normalizing data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_test_real = scaler.transform(X_test_real)

#%% 
#Tuning hyperparameters "Kernel", "C", and "gamma".

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

#Using GridSearchCV to search for best parameterset.

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)
    
    
    print()
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
# %%
#define the classifer.
svcClassifier = SVC(kernel = 'rbf', C = 1000, gamma = 0.001,  random_state= 42)
svcClassifier.fit(X_train, y_train)

y_pred = svcClassifier.predict(X_test_real)
print(confusion_matrix(y_test_real,y_pred))
print(classification_report(y_test_real,y_pred))
print(accuracy_score(y_test_real, y_pred))
print(average_precision_score(y_test_real, y_pred))


title = 'Normalized confusion matrix using SVC and PCA'
disp = plot_confusion_matrix(svcClassifier, X_test_real, y_test_real,
                                display_labels=['Weak Inhibitor', 'Strong Inhibitor'],
                                cmap=plt.cm.Blues,
                                normalize='true')
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)

plt.show()
plt.savefig('figures/SVC_Cmatrix_pca.png')


# %%
