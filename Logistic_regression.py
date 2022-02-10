#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing as prep
from sklearn.model_selection import GridSearchCV
# %%
# Import data
data = pd.read_csv('data/Unique_features.csv', index_col = 'ID')
df_out = pd.read_csv('data/features_out_cat.csv', index_col= 'ID')
pca_data = pd.read_csv('data/pca_scores.csv', index_col= 'ID')
D3_pca_data = pd.read_csv('data/D3C_pca_scores.csv', index_col = 'Cmpd_ID')
D3_data = pd.read_csv('data/D3C_clean_features.csv', index_col= 'Cmpd_ID')
# %%

X = data.loc[:, 'MolLogP':'VSA_EState10']
X_pca = pca_data
y = df_out['Binary_value']

X_test_real = D3_data.loc[:, 'MolLogP':'VSA_EState10']
X_test_real_pca = D3_pca_data
y_test_real = D3_data.loc[:, 'pIC50']

for i in range(y_test_real.shape[0]):
    if y_test_real[i] >=7:
       y_test_real[i] = 1
        #print('YES!!!!')
    else:
       y_test_real[i] = 0
#y_test_real
X

# %%
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.5, random_state = 42)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_pca, y, test_size = 0.3, random_state = 42)

print('Ratio number of instances in the Test/ Train: {}'.format(len(y_test)/len(y_train)))

print('Ratio of 1 in Train set: {}'.format(y_train.sum()/len(y_train)))

print('Ratio of 1 in Test set: {}'.format(y_test.sum()/len(y_test)))

#%%

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train1 = scaler.transform(X_train1)
X_test1 = scaler.transform(X_test1)

#X_test_real = scaler.transform(X_test_real)
X_test_real_pca = scaler.transform(X_test_real_pca)

# %%
#Tuning Hyper parameters using GridCv

tuned_parameters = [{'penalty': ['l1'], 'C': [1, 10, 100, 1000]},
                    {'penalty': ['l2'], 'C': [1, 10, 100, 1000]}]

scores = ['precision']


# %%
#UsingGridCV to tune penalty and C.
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        LogisticRegression(solver = 'liblinear'), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)

    

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

#%%
scores_data = pd.DataFrame()
scores_data['L1_scores'] = clf.cv_results_['mean_test_score'].reshape(4,2, order = 'F')[:,0]
scores_data['L2_scores'] = clf.cv_results_['mean_test_score'].reshape(4,2, order = 'F')[:,1]
scores_data['L1_stdscores'] = clf.cv_results_['std_test_score'].reshape(4,2, order = 'F')[:,0]
scores_data['L2_stdscores'] = clf.cv_results_['std_test_score'].reshape(4,2, order = 'F')[:,1]
scores_data.head()
Clog = [0,1,2,3]
fig, ax = plt.subplots()
#sns.lineplot(x =Clog , y = 'L1_scores',hue = 'event',  err_style = 'bars',ci = 68,  markers=True, dashes=False, data = scores_data,)
#sns.lineplot(x =Clog , y = 'L2_scores', ci = 'L2_stdscores', err_style = 'band', data = scores_data)
ax.plot(Clog, scores_data['L1_scores'],  sns.xkcd_rgb["pale red"],  label ='L1:Lasso' )
ax.plot(Clog, scores_data['L2_scores'],  sns.xkcd_rgb["medium green"],  label = 'L2:Ridge' )
plt.ylim(0,1)
plt.xlabel('C')
plt.ylabel('Precision score')
ax.legend()
plt.title('Cross validation scores for Log reg and PCA using GridsearchCV')
#ax.set_facecolor('xkcd:pale gray')
plt.savefig('figures/tuning_param_log_reg_recall.png')
#%%

log_regClassifier = LogisticRegression (penalty = 'l1', C = 1, solver = 'liblinear')
log_regClassifier.fit(X_train1, y_train1)

y_pred = log_regClassifier.predict(X_test_real_pca)

print(confusion_matrix(y_test_real, y_pred))
print(classification_report(y_test_real, y_pred))

#%%

title = 'Normalized confusion matrix using Log Reg and PCA'
disp = plot_confusion_matrix(log_regClassifier, X_test_real_pca, y_test_real,
                                display_labels=['Weak Inhibitor', 'Strong Inhibitor'],
                                cmap=plt.cm.Blues,
                                normalize='true')
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)

plt.show()
plt.savefig('figures/LogReg_Cmatrix.png', format= 'png')

# %%
