#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix
import numpy as np


# In[2]:


#initiate scoring data frame
test_scores = {'file_train':[], 'file_test':[], 'best_parameter':[], 'accuracy':[],
               'precision':[], 'recall':[], 'f1':[], 'mcc':[]}

#necessary functions
def load_train_file(file):
    df = pd.read_csv(file)

    #Train data
    X = df[df.columns[1:-6]]
    X = X.astype('float64')

    #define y
    y = df.diagnosis
    print('Fraction of Infected: ', sum(y)/len(y))
    print(X.shape)
    print(y.shape)
    return X, y

def load_test_file(file):
    df = pd.read_csv(file)

    #Train data
    X = df[df.columns[1:-6]]
    X = X.astype('float64')

    #define y
    y = df.diagnosis
    print('Fraction of Infected: ', sum(y)/len(y))
    print(X.shape)
    print(y.shape)
    return X, y

def load_val_file(file):
    df = pd.read_csv(file)

    #Train data
    X = df[df.columns[1:-6]]
    X = X.astype('float64')

    #define y
    y = df.diagnosis
    print('Fraction of Infected: ', sum(y)/len(y))
    print(X.shape)
    print(y.shape)
    return X, y

#define mathew correlation cofficient (MCC)
def mcc(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    if np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) == 0:
        return 0
    else:
        mcc_score = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return mcc_score

#SVM RBF Hyperparameter Optimization - optimizes on accuracy score and returns best parameter in dictionary format
def SVC_rbf_best_param_acc(C_range, gamma_range, X_train, y_train, X_val, y_val):
    rec = {'param':[], 'score':[]}
    pg = ParameterGrid({'C': C_range, 'gamma': gamma_range})
    for i in pg:
        print("Starting", i)
        model = SVC(kernel='rbf', C=i['C'], gamma=i['gamma'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        rec['param'].append(i)
        rec['score'].append(accuracy)
        print("Ended", i)
    df = pd.DataFrame(rec)
    highest_score_index = df['score'].idxmax()
    best_parameter = df['param'][highest_score_index]
    return best_parameter

#SVM RBF Hyperparameter Optimization - optimizes on f1 score and returns best parameter in dictionary format
def SVC_rbf_best_param_f1(C_range, gamma_range, X_train, y_train, X_val, y_val):
    rec = {'param':[], 'score':[]}
    pg = ParameterGrid({'C': C_range, 'gamma': gamma_range})
    for i in pg:
        print("Starting", i)
        model = SVC(kernel='rbf', C=i['C'], gamma=i['gamma'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        rec['param'].append(i)
        rec['score'].append(f1)
        print("Ended", i)
    df = pd.DataFrame(rec)
    highest_score_index = df['score'].idxmax()
    best_parameter = df['param'][highest_score_index]
    return best_parameter

#SVM Linear Hyperparameter Optimization - optimizes on f1 score and returns best parameter in dictionary format
def SVC_lin_best_param_f1(C_range, X_train, y_train, X_val, y_val):
    rec = {'param':[], 'score':[]}
    pg = ParameterGrid({'C': C_range})
    for i in pg:
        print("Starting", i)
        model = SVC(kernel='linear', C=i['C'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        rec['param'].append(i)
        rec['score'].append(f1)
        print("Ended", i)
    df = pd.DataFrame(rec)
    highest_score_index = df['score'].idxmax()
    best_parameter = df['param'][highest_score_index]
    return best_parameter

#SVM Linear Hyperparameter Optimization - optimizes on accuracy score and returns best parameter in dictionary format
def SVC_lin_best_param_acc(C_range, X_train, y_train, X_val, y_val):
    rec = {'param':[], 'score':[]}
    pg = ParameterGrid({'C': C_range})
    for i in pg:
        print("Starting", i)
        model = SVC(kernel='linear', C=i['C'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        rec['param'].append(i)
        rec['score'].append(accuracy)
        print("Ended", i)
    df = pd.DataFrame(rec)
    highest_score_index = df['score'].idxmax()
    best_parameter = df['param'][highest_score_index]
    return best_parameter

#RF Hyperparameter Optimization - optimizes on accuracy score and returns best parameter in dictionary format
def rf_best_param_acc(max_depth, min_samples_leaf, X_train, y_train, X_val, y_val):
    rec = {'param':[], 'score':[]}
    pg = ParameterGrid({"max_depth":max_depth, 'min_samples_leaf': min_samples_leaf})
    for i in pg:
        print("Starting", i)
        model = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=2,
                                       max_features='sqrt', max_leaf_nodes=None, 
                                       max_depth=i['max_depth'], min_samples_leaf=i['min_samples_leaf'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        rec['param'].append(i)
        rec['score'].append(accuracy)
        print("Ended", i)
    df = pd.DataFrame(rec)
    highest_score_index = df['score'].idxmax()
    best_parameter = df['param'][highest_score_index]
    return best_parameter

#RF Hyperparameter Optimization - optimizes on f1 score and returns best parameter in dictionary format
def rf_best_param_f1(max_depth, min_samples_leaf, X_train, y_train, X_val, y_val):
    rec = {'param':[], 'score':[]}
    pg = ParameterGrid({"max_depth":max_depth, 'min_samples_leaf': min_samples_leaf})
    for i in pg:
        print("Starting", i)
        model = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=2,
                                       max_features='sqrt', max_leaf_nodes=None, 
                                       max_depth=i['max_depth'], min_samples_leaf=i['min_samples_leaf'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        rec['param'].append(i)
        rec['score'].append(f1)
        print("Ended", i)
    df = pd.DataFrame(rec)
    highest_score_index = df['score'].idxmax()
    best_parameter = df['param'][highest_score_index]
    return best_parameter

#Builds SVM RBF model on best parameter, fits on test data and records and returns scoring matrics
def SVC_rbf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file):
    test_scores['file_train'].append(train_file)
    test_scores['file_test'].append(test_file)
    print("Starting fitting on best_parameter", best_parameter)
    C = best_parameter['C']
    gamma = best_parameter['gamma']
    model = SVC(kernel='rbf', C=C, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #scoring
    target_names = ['Healthy', 'Infected']
    rep = classification_report(y_test, y_pred, target_names=target_names)
    print(rep)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mcc_score = mcc(y_test, y_pred)

    #record and export
    test_scores['best_parameter'].append(best_parameter)
    test_scores['accuracy'].append(accuracy)
    test_scores['precision'].append(precision)
    test_scores['recall'].append(recall)
    test_scores['f1'].append(f1)
    test_scores['mcc'].append(mcc_score)
    df = pd.DataFrame(test_scores)
    df.to_csv("scores.csv")
    print("Finished fitting on best_parameter", best_parameter)

#Builds SVM Linear model on best parameter, fits on test data and records and returns scoring matrics
def SVC_lin_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file):
    test_scores['file_train'].append(train_file)
    test_scores['file_test'].append(test_file)
    print("Starting fitting on best_parameter", best_parameter)
    C = best_parameter['C']
    model = SVC(kernel='linear', C=C)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #scoring
    target_names = ['Healthy', 'Infected']
    rep = classification_report(y_test, y_pred, target_names=target_names)
    print(rep)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mcc_score = mcc(y_test, y_pred)

    #record and export
    test_scores['best_parameter'].append(best_parameter)
    test_scores['accuracy'].append(accuracy)
    test_scores['precision'].append(precision)
    test_scores['recall'].append(recall)
    test_scores['f1'].append(f1)
    test_scores['mcc'].append(mcc_score)
    df = pd.DataFrame(test_scores)
    df.to_csv("scores.csv")
    print("Finished fitting on best_parameter", best_parameter)

#Builds RF model on best parameter, fits on test data and records and returns scoring matrics
def rf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file):
    test_scores['file_train'].append(train_file)
    test_scores['file_test'].append(test_file)
    print("Starting fitting on best_parameter", best_parameter)
    max_depth = best_parameter['max_depth']
    min_samples_leaf = best_parameter['min_samples_leaf']
    model = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=2,
                                       max_features='sqrt', max_leaf_nodes=None, 
                                       max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #scoring
    target_names = ['Healthy', 'Infected']
    rep = classification_report(y_test, y_pred, target_names=target_names)
    print(rep)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mcc_score = mcc(y_test, y_pred)

    #record and export
    test_scores['best_parameter'].append(best_parameter)
    test_scores['accuracy'].append(accuracy)
    test_scores['precision'].append(precision)
    test_scores['recall'].append(recall)
    test_scores['f1'].append(f1)
    test_scores['mcc'].append(mcc_score)
    df = pd.DataFrame(test_scores)
    df.to_csv("scores.csv")
    print("Finished fitting on best_parameter", best_parameter)    


# In[ ]:


#SVM Linear
C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]

#load training and val files
train_file = 'phyla_all_3949x1177_pmi_0_clr_75p_MAD.csv'
val_file = 'phyla_all_565x1177_pmi_0_clr_10p_MAD.csv'

X_train, y_train = load_train_file(train_file)
X_val, y_val = load_val_file(val_file)

#best parameters on accuracy
best_parameter = SVC_lin_best_param_acc(C_range, X_train, y_train, X_val, y_val)
print(best_parameter)

#load test file
test_file = 'phyla_all_753x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_lin_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_lin_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_lin_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)


# In[3]:


#SVM Linear
C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]

#load training and val files
train_file = 'phyla_biopsy_1109x1177_pmi_0_clr_75p_MAD.csv'
val_file = 'phyla_biopsy_164x1177_pmi_0_clr_10p_MAD.csv'

X_train, y_train = load_train_file(train_file)
X_val, y_val = load_val_file(val_file)

#best parameters on f1
best_parameter = SVC_lin_best_param_f1(C_range, X_train, y_train, X_val, y_val)
print(best_parameter)

#load test file
test_file = 'phyla_all_753x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_lin_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_lin_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_lin_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)


# In[4]:


#SVM Linear
C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]

#load training and val files
train_file = 'phyla_stool_2840x1177_pmi_0_clr_75p_MAD.csv'
val_file = 'phyla_stool_401x1177_pmi_0_clr_10p_MAD.csv'

X_train, y_train = load_train_file(train_file)
X_val, y_val = load_val_file(val_file)

#best parameters on f1
best_parameter = SVC_lin_best_param_f1(C_range, X_train, y_train, X_val, y_val)
print(best_parameter)

#load test file
test_file = 'phyla_all_753x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_lin_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_lin_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_lin_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)


# In[9]:


#RF
max_depth = list(range(1,11))
min_samples_leaf = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]

#load training and val files
train_file = 'phyla_all_3949x1177_pmi_0_clr_75p_MAD.csv'
val_file = 'phyla_all_565x1177_pmi_0_clr_10p_MAD.csv'

X_train, y_train = load_train_file(train_file)
X_val, y_val = load_val_file(val_file)

#best parameters on acc
best_parameter = rf_best_param_acc(max_depth, min_samples_leaf, X_train, y_train, X_val, y_val)
print(best_parameter)

#load test file
test_file = 'phyla_all_753x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
rf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
rf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
rf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)


# In[5]:


#RF
max_depth = list(range(1,11))
min_samples_leaf = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]

#load training and val files
train_file = 'phyla_biopsy_1109x1177_pmi_0_clr_75p_MAD.csv'
val_file = 'phyla_biopsy_164x1177_pmi_0_clr_10p_MAD.csv'

X_train, y_train = load_train_file(train_file)
X_val, y_val = load_val_file(val_file)

#best parameters on f1
best_parameter = rf_best_param_f1(max_depth, min_samples_leaf, X_train, y_train, X_val, y_val)
print(best_parameter)

#load test file
test_file = 'phyla_all_753x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
rf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
rf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
rf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)


# In[6]:


#RF
max_depth = list(range(1,11))
min_samples_leaf = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]

#load training and val files
train_file = 'phyla_stool_2840x1177_pmi_0_clr_75p_MAD.csv'
val_file = 'phyla_stool_401x1177_pmi_0_clr_10p_MAD.csv'

X_train, y_train = load_train_file(train_file)
X_val, y_val = load_val_file(val_file)

#best parameters on f1
best_parameter = rf_best_param_f1(max_depth, min_samples_leaf, X_train, y_train, X_val, y_val)
print(best_parameter)

#load test file
test_file = 'phyla_all_753x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
rf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
rf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
rf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)


# In[10]:


#SVM RBF
C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]
gamma_range = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

#load training and val files
train_file = 'phyla_all_3949x1177_pmi_0_clr_75p_MAD.csv'
val_file = 'phyla_all_565x1177_pmi_0_clr_10p_MAD.csv'

X_train, y_train = load_train_file(train_file)
X_val, y_val = load_val_file(val_file)

#best parameters on acc
best_parameter = SVC_rbf_best_param_acc(C_range, gamma_range, X_train, y_train, X_val, y_val)
print(best_parameter)

#load test file
test_file = 'phyla_all_753x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_rbf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_rbf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_rbf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)


# In[7]:


#SVM RBF
C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]
gamma_range = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

#load training and val files
train_file = 'phyla_biopsy_1109x1177_pmi_0_clr_75p_MAD.csv'
val_file = 'phyla_biopsy_164x1177_pmi_0_clr_10p_MAD.csv'

X_train, y_train = load_train_file(train_file)
X_val, y_val = load_val_file(val_file)

#best parameters on f1
best_parameter = SVC_rbf_best_param_f1(C_range, gamma_range, X_train, y_train, X_val, y_val)
print(best_parameter)

#load test file
test_file = 'phyla_all_753x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_rbf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_rbf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_rbf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)


# In[8]:


#SVM RBF
C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]
gamma_range = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

#load training and val files
train_file = 'phyla_stool_2840x1177_pmi_0_clr_75p_MAD.csv'
val_file = 'phyla_stool_401x1177_pmi_0_clr_10p_MAD.csv'

X_train, y_train = load_train_file(train_file)
X_val, y_val = load_val_file(val_file)

#best parameters on f1
best_parameter = SVC_rbf_best_param_f1(C_range, gamma_range, X_train, y_train, X_val, y_val)
print(best_parameter)

#load test file
test_file = 'phyla_all_753x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_rbf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_rbf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)

#load test file
test_file = 'phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv'
X_test, y_test = load_test_file(test_file)

#fit and score on test file
print('Scores on ', test_file)
SVC_rbf_test_score(test_scores, best_parameter, X_train, y_train, X_test, y_test, train_file, test_file)


# In[ ]:


#shut down the system once the process is complete
import os
os.system("shutdown /s /t 1")


# In[ ]:




