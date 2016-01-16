# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:57:38 2015

@author: amoussoubaruch
"""
##############################################################################
##############################################################################

    # Score de délivrance d’un brevet européen
    # Challenge caisse de dépôt
    # Auteur : Baruch AMOUSSOU-DJANGBAN

##############################################################################
##############################################################################

##########                   Import packages                       ##########  
import csv
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

##Missing values
from sklearn.preprocessing import Imputer
from sklearn.base import TransformerMixin


##########                  Load  Data Train                       ########## 
#data_train=pd.read_csv("https://dl.dropboxusercontent.com/u/2140486/mdi343_2015/train.csv", delimiter=';')
data_train=pd.read_csv("/Users/amoussoubaruch/Documents/Challenge_caisse_depot/DataSet/train.csv", delimiter=';')



##########                  Load  Data Tesr                       ########## 
#data_test=pd.read_csv("https://dl.dropboxusercontent.com/u/2140486/mdi343_2015/test.csv", delimiter=';')
data_test=pd.read_csv("/Users/amoussoubaruch/Documents/Challenge_caisse_depot/DataSet/test.csv", delimiter=';')

##########                   data Pre Processing                    ########## 
# Shape data_train
n,p = data_train.shape

"""
We have 259431 exemples and 50 features
"""
## features names
column_name = data_train.columns.values.tolist()

## Type of  features names
data_train.dtypes

## Head data
data_50= data_train.head(50)
##Save 50 row in cvs 
data_50.to_csv("/Users/amoussoubaruch/Documents/Challenge_caisse_depot/sortie.csv",sep=";", index=False)

## Tail data
data_train.tail()

## Missing Values
data_train.isnull().sum()

##List of feature have lot of missing value
liste_lot_of_missing = []

for c in data_train.columns.values:
    if data_train[c].isnull().sum() > n/2.:
        liste_lot_of_missing.append(c)

## Delete  liste_lot_of_missing on column_name
variables = list(set(column_name) - set(liste_lot_of_missing))

## Select colomns contain month
data_cols = [col for col in data_train.columns if 'MONTH' in col]
#Delete fisrt element o list
data_cols.pop(0)

## Slicing column date to recover year and month in different columns
for c in data_cols:
    print c
    data_train[c+'year'] = data_train[c].str[3:]
    data_train[c+'month'] = data_train[c].str[:2]

for c in data_cols:
    print c
    data_test[c+'year'] = data_test[c].str[3:]
    data_test[c+'month'] = data_test[c].str[:2]

## variables that we select
variables1 = data_train.columns.values.tolist()
variables = list(set(variables1) - set(data_cols))
len(variables)


### Data train final
data_train_final = data_train[variables]
data_train_final.columns.values.tolist()

## Data Tests
variab = [x for x in variables if x != 'VARIABLE_CIBLE']
data_test_final = data_test[variab]
data_test.columns.values.tolist()

## Missing Values
data_train_final.isnull().sum()
data_train_final.dtypes

## Head data
data_50= data_train_final.head(50)
##Save 50 row in cvs 
data_50.to_csv("/Users/amoussoubaruch/Documents/Challenge_caisse_depot/sortie.csv",sep=";", index=False)


## Fill missing value
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
        
## Data_train fill na
data_train_fin = DataFrameImputer().fit_transform(data_train_final)
data_train_fin.dtypes

## Data_test Fill na
data_test_fin = DataFrameImputer().fit_transform(data_test_final)
        
## Missing Values data_train1
data_train_fin.isnull().sum()



#############################
##########                   Feature selecton                   ########## 
#df[['two', 'three']] = df[['two', 'three']].astype(float)
                                             #############################
from sklearn.feature_selection import SelectKBest , f_classif, chi2

##Select cible variable
y_train = data_train_fin.VARIABLE_CIBLE == 'GRANTED'
plt.hist(y_train)
############################
####### QUANTITAIVE VARIABLE
############################
##List variable quantitative int
liste_int = []

for c in data_train_fin.columns:
    if data_train_fin[c].dtype == np.int64:
    ##if isinstance(data_train[c], np.int64) == True:
        liste_int.append(c)
    
##List variable qualitative float
liste_float = []

for c in data_train_fin.columns.values:
    if data_train_fin[c].dtype == np.float64:
        liste_float.append(c) 

##list quantitative variable
liste_quanti =  liste_int  + liste_float
len(liste_quanti)

###### ANOVA TEST QUANTI
X_quanti = SelectKBest(k=10).fit(data_train_fin[liste_quanti], y_train)
indice_quanti = X_quanti.get_support(indices=True)

## data_train_quanti
data_train_quanti = data_train_fin[liste_quanti]
quanti_list = data_train_quanti.columns.values.tolist()
## Choose variables with indices
from operator import itemgetter
##Select variables quanti
quanti_select =list(itemgetter(*indice_quanti.tolist())(quanti_list))

quanti_select = liste_quanti
data_train_quanti = data_train_fin[quanti_select]
data_test_quanti = data_test_fin[quanti_select]



############################
####### QUALITATIVE VARIABLE
############################

### Function to compute chi test

#def categories(series):
#    return range(int(series.min()), int(series.max()) + 1)
#
#
#def chi_square_of_df_cols(df, col1, col2):
#    df_col1, df_col2 = df[col1], df[col2]
#
#    result = [[sum((df_col1 == cat1) & (df_col2 == cat2))
#               for cat2 in categories(df_col2)]
#              for cat1 in categories(df_col1)]
#    return scs.chi2_contingency(result)
    
    
##List qualitative
liste_quali = []

for c in data_train_fin.columns.values:
    if data_train_fin[c].dtype == np.object:
        liste_quali.append(c) 
        
## Remove cible
liste_quali = [x for x in liste_quali if x != 'VARIABLE_CIBLE']

len(liste_quali)

#### CHI TEST QUALI
#X_quali = SelectKBest(chi2, k=10).fit(data_train_fin[liste_quali], y_train)
#indice_quali = X_quanti.get_support(indices=True)

def chisq_of_df_cols(df, c1, c2):
    groupsizes = df.groupby([c1, c2]).size()
    ctsum = groupsizes.unstack(c1)
    # fillna(0) is necessary to remove any NAs which will cause exceptions
    return(scs.chi2_contingency(ctsum.fillna(0)))[1]

dico_chi2 = {}

for var in liste_quali:
    print var
    chi2 = chisq_of_df_cols(data_train_fin, var, 'VARIABLE_CIBLE')
    dico_chi2[var]= chi2
  
table=pd.DataFrame(dico_chi2.items(), columns=['Variables', 'P_Value']) 

table = table.sort(['P_Value'], ascending= False)

quali_select =table.head(2)['Variables'].unique().tolist()

quali_select = liste_quali

data_train_quali = data_train_fin[quali_select]
data_test_quali = data_test_fin[quali_select]

data_train_quali1 = None
data_test_quali1 = None

for c in quali_select:
    res = pd.get_dummies(data_train_quali[c])
    data_train_quali1 =pd.concat([data_train_quali1, res], axis=1)

for c in quali_select:
    res = pd.get_dummies(data_test_quali[c])
    data_test_quali1 =pd.concat([data_test_quali1, res], axis=1)
    

################## TRAIN FINAL
Data_Trainn = pd.concat([data_train_quanti, data_train_quali1], axis=1)
################ TEST FINAL
Data_Testt = pd.concat([data_test_quanti, data_test_quali1], axis=1)


################## TRAIN FINAL
Data_Trainn = data_train_quanti
################ TEST FINAL
Data_Testt = data_test_quanti


#############################
##########                   Classifier                   ########## 
##
X_train = Data_Trainn.values

X_test = Data_Testt.values

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=10))
clf=clf.fit(X_train, y_train)

#clf = RandomForestClassifier()
#clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)[:, 1]
np.savetxt('/Users/amoussoubaruch/Documents/BaruchAMOUSSOU/challenge_gearud/y_pred.txt', y_pred, fmt='%s')

### Variable type
data_train.dtypes

##Liste variable quantitative int
liste_int = []

for c in data_train.columns:
    if data_train[c].dtype == np.int64:
    ##if isinstance(data_train[c], np.int64) == True:
        liste_int.append(c)
    
##Liste variable qualitative float
liste_float = []

for c in data_train.columns.values:
    if data_train[c].dtype == np.float64:
        liste_float.append(c) 

##liste quantitative variable
liste_quanti =  liste_int  = liste_float
len(liste_quanti)
feature_names = liste_quanti

##
X_train = data_train[feature_names].values

X_test = data_test[feature_names].values
y_test = data_test.VARIABLE_CIBLE 
data_test.columns.values


### Logistic regression
from sklearn.linear_model import LogisticRegression
imputer = Imputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

model = LogisticRegression(C=0.7)
model = model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
np.savetxt('/Users/amoussoubaruch/Documents/BaruchAMOUSSOU/challenge_gearud/y_pred.txt', proba, fmt='%s')



from sklearn.tree import tree
imputer = Imputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


profondeur_maxi=range(10,25)
dicoarbre={}
for a in profondeur_maxi:
    clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=a).fit(X_train, y_train)
    scores=clf.predict_proba(X_test)[:, 1]  
    dicoarbre[a]=scores.min()
table_arbre=pd.DataFrame(dicoarbre.items(), columns=['profondeur', 'score']) 




y_pred_train = clf.predict_proba(X_train)[:, 1]

print('Score (optimiste) sur le train : %s' % roc_auc_score(y_train, y_pred_train))

y_pred = clf.predict_proba(X_test)[:, 1]
np.savetxt('/Users/amoussoubaruch/Documents/BaruchAMOUSSOU/challenge_gearud/y_pred.txt', y_pred, fmt='%s')


##Liste variable qualitative
liste_str = []

for c in data_train.columns.values:
    if data_train[c].dtype == np.object:
        liste_str.append(c)     

### Variables 
data_train['VOIE_DEPOT']   


### AdaBoost
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=10))
clf=clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)[:, 1]
np.savetxt('/Users/amoussoubaruch/Documents/BaruchAMOUSSOU/challenge_gearud/y_pred.txt', y_pred, fmt='%s')


### Transform in variable dummy
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
r = enc.fit(data_train['VOIE_DEPOT'])
data_train['VOIE_DEPOT'].unique()


### Test modelisation
y_train = data_train_fin.VARIABLE_CIBLE 

##
feature_names = [x for x in variables if x != 'VARIABLE_CIBLE']
len(feature_names)

X_train = data_train_fin[feature_names].values

X_test = data_test_fin[feature_names].values

##Arbre


from sklearn.tree import tree
imputer = Imputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


profondeur_maxi=range(10,25)
dicoarbre={}
for a in profondeur_maxi:
    clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=10).fit(X_train, y_train)
    scores=clf.predict_proba(X_test)[:, 1]  
    dicoarbre[a]=scores.min()
table_arbre=pd.DataFrame(dicoarbre.items(), columns=['profondeur', 'score']) 



y_pred = clf.predict_proba(X_test)[:, 1]
np.savetxt('/Users/amoussoubaruch/Documents/BaruchAMOUSSOU/challenge_gearud/y_pred.txt', y_pred, fmt='%s')