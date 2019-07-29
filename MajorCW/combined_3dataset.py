#!/usr/bin/env python
# coding: utf-8

# In[1]:


# coding: utf-8
log = print

## import library
import gzip
import math
import numpy as np
from sklearn import linear_model, metrics, ensemble
import pandas as pd
import h5py
import tensorflow as tf
import random
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn import svm, datasets
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import logging
import argparse
import os


# In[2]:


sourcedata = pd.read_csv('HiSeqV2_PANCAN.gz', compression='gzip', delim_whitespace = True)

sourcedata2 = pd.read_csv('HiSeqV2_PANCAN_Z.gz', compression='gzip', delim_whitespace = True)


# In[5]:


# data_clean = sourcedata.select_dtypes(include=['float64'])
# column_names = list(data_clean.columns.values)
# label = [s[-2:] for s in column_names]
# label = [0 if s in ('01', '02', '03', '04', '05', '06', '07', '08', '09') else 1 for s in label]

print (sourcedata.info())
print (sourcedata2.info())


# In[ ]:





# In[23]:


data1 = sourcedata.T

data2 = sourcedata2.T


# In[ ]:


print(data1)


# In[38]:


print(data2)


# In[7]:


data1.columns = data1.iloc[0]
data2.columns = data2.iloc[0]
data1 = data1[1:]
data2 = data2[1:]


# In[48]:


print(data1)


# In[43]:


print(data2)


# In[24]:


print((data1.index))


def change_index(data):
    p = data.index

    index_list = []
    for i in data.index:
        index_list.append(i[-2:])
    print((index_list))
    return index_list
index_list = change_index(data1)


# In[25]:


data1.index = [0 if s in ('01', '02', '03', '04', '05', '06', '07', '08', '09') else 1 for s in index_list]


# In[26]:


index_list2 = change_index(data2)


# In[27]:



data2.index = [2 if s in ('01', '02', '03', '04', '05', '06', '07', '08', '09') else 1 for s in index_list2]


# In[28]:


print(data1)


# In[29]:


print(data2)


# In[77]:


print(data2)


# In[30]:


data1.columns = data1.iloc[0]
data2.columns = data2.iloc[0]


# In[31]:


data1 = data1[1:]
data2 = data2[1:]


# In[32]:


print(data1)


# In[33]:


print(data2)


# In[34]:


labels = data1.columns


# In[270]:


print(labels)


# In[36]:


data2 = data2[labels]


# In[92]:


print(data2)


# In[37]:


frames = [data1,data2]


# In[ ]:


result = pd.concat(frames)
print(result)


# In[39]:


result.info()


# In[268]:


sick = result.index.values 


# In[269]:


print((sick))


# In[41]:



sick = sick.tolist()


# In[42]:



print (sick.count(0)) 
print (sick.count(1)) 
print (sick.count(2)) 


# In[125]:


sizes = [1104, 165, 383]
labels = ["breast invasive carcinoma", "Non-Cancer", "colon & rectum adenocarcinoma "]

plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.title('Data Balance Analysis')
plt.show()


# In[43]:


def plot_coefficients(classifier, feature_names, top_features=20):
    #coef = classifier.coef_.ravel()
    coef = classifier.feature_importances_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    print(feature_names[top_coefficients])
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.xlabel('Top important Feature')
    plt.ylabel('Coefficients Of Feature')
    plt.title('LDA')
    plt.show()
    


# In[256]:


result.iloc[0]


# In[129]:


result.shape


# In[259]:


train_index = random.sample(range(0, 1652), 1320)


# In[260]:


train_x = [result.iloc[j] for j in train_index]
train_y = [sick[j] for j in train_index]
test_x = [result.iloc[j] for j in range(0, 1652) if j not in train_index]
test_y = [sick[j] for j in range(0, len(sick)) if j not in train_index]


# In[261]:


train_y.count(1)
#train_y.count(0)


# In[262]:


#LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(n_components= 100)
clf.fit(train_x, train_y)
probas = clf.predict(test_x)

precision_recall_fscore_support(test_y, probas,average='macro')


# In[ ]:


tree_clf = DecisionTreeClassifier()
tree_clf.fit(train_x, train_y)


# In[ ]:


probas = tree_clf.predict(test_x)


# In[ ]:


precision_recall_fscore_support(test_y, probas,average='macro')[2]


# In[274]:


def filter_tree(x, y,tx,ty):
    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(x, y)
    probas = tree_clf.predict(tx)
    return precision_recall_fscore_support(ty, probas,average='macro')[2]


# In[276]:


NOF = []
SF =[]
number  = 0
for i in labels:
    SF.append(i)
    number = number + 1
    x = [result[SF].iloc[j] for j in train_index]
    y = [sick[j] for j in train_index]
    tx = [result[SF].iloc[j] for j in range(0, 1652) if j not in train_index]
    ty = [sick[j] for j in range(0, len(sick)) if j not in train_index]
    f1 = filter_tree(x, y, tx,ty)
    NOF.append(f1) 
    print(number)
    if number > 400:
        break


# In[286]:


NOF.sort(reverse = False)


# In[283]:





# In[287]:


plt.plot(NOF)
plt.xlabel('Top important Feature')
plt.ylabel('F1 score')
plt.title('DT')
plt.show()


# In[84]:


clf.xbar_


# In[85]:


clf.classes_


# In[130]:


plot_coefficients(clf, p)


# In[66]:


from sklearn.decomposition import PCA
X = train_x
pca = PCA(n_components=100)
pca.fit(X)


# In[63]:


pca.explained_variance_ratio_


# In[69]:


plot_coefficients(pca, p)


# In[138]:


def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    #coef = classifier.feature_importances_.ravel()
    #coef = classifier.explained_variance_ratio_
    #coef = classifier.xbar_
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    #print(top_positive_coefficients)
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    #print(top_coefficients)
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    print(feature_names[top_coefficients])
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.xlabel('Top important Feature')
    plt.ylabel('Coefficients Of Feature')
    plt.title('DT')
    plt.show()


# In[61]:


print(result.columns)


# In[60]:


p = (result.columns).tolist()


# In[ ]:


plot_coefficients(clf, p)


# In[325]:


probas = clf.predict(test_x)
precision_recall_fscore_support(test_y, probas,average='macro')


# In[ ]:


X = result.T

y = result.columns


# In[ ]:


print(X)


# In[220]:





# In[92]:


from sklearn.metrics import precision_recall_fscore_support
X = result

y = np.array(sick)


random_state = np.random.RandomState(0)

cv = StratifiedKFold(n_splits = 2)
logreg = linear_model.LogisticRegression(penalty = 'l1', C=1e5, 
                                        random_state = random_state)

f1_scores = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
print(len(X))
print(len(y))


# In[235]:


for train, test in cv.split(X, y):
    probas = logreg.fit(X.iloc[train], y[train]).predict(X.iloc[test])
    #fpr, tpr, thresholds = roc_curve(y[test], probas)
    #f1_score = metrics.f1_score(y[test], probas)
    #tprs.append(interp(mean_fpr, fpr, tpr))
    #f1_scores.append(f1_score)
    #tprs[-1][0] = 0.0
    f1_score =  precision_recall_fscore_support(y[test], probas,average='macro')
    print(list(f1_score))
    #roc_auc = auc(fpr, tpr)
    #aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw = 1, alpha = 0.3, label = "ROC fold %d (AUC = %0.2f)" % (i, roc_auc))
    i += 1


# In[244]:


from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier()
tree_clf.fit(train_x, train_y)


# In[113]:


test_y.count(1)


# In[ ]:


probas = tree_clf.predict(test_x)
precision_recall_fscore_support(test_y, probas,average='macro')


# In[246]:


mean_squared_error(test_y, probas)


# In[121]:


plot_coefficients(tree_clf, p)


# In[116]:


from sklearn.tree import export_graphviz
export_graphviz(tree_clf,
                out_file="tree.dot",
                feature_names=p,
                rounded=True,
                filled=True
               )


# In[117]:


get_ipython().system('dot -Tpng tree.dot -o tree.png')
from IPython.display import Image
Image(filename='tree.png')


# In[135]:


#SVM

random_state = np.random.RandomState(0)
new_svm =  svm.SVC(kernel = 'linear', probability = True, )
new_svm.fit(train_x, train_y)


# In[139]:


plot_coefficients(new_svm, p)


# In[237]:


random_forest_reg = ensemble.RandomForestRegressor()
random_forest_reg.fit(train_x, train_y)

##plot the ROC curve for the training set and compute the auc_score
training_set_z_rf = random_forest_reg.predict(test_x)
# probas = random_forest_reg.predict(test_x)
# precision_recall_fscore_support(test_y, probas,average='macro')
#fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(test_y, training_set_z_rf)
#training_auc_score_rf = metrics.roc_auc_score(test_y, training_set_z_rf)
# plt.plot(fpr_rf, tpr_rf)
# plt.title("random_forest_test_roc")
# plt.xlabel("false positive rate")
# plt.ylabel("true positive rate")
# plt.show()
#print("Random Forest Training AUC: " + str(training_auc_score_rf))


# In[128]:


plot_coefficients(random_forest_reg, p)


# In[238]:


probas = random_forest_reg.predict(test_x)
#precision_recall_fscore_support(test_y, probas,average='macro')


# In[239]:


precision_recall_fscore_support(test_y, probas,average='macro')


# In[316]:


# trainx = np.array(train_x)
# trainy = np.array(train_y)
# testx = np.array(test_x)
# testy = np.array(test_y)

# trainy = trainy.reshape((trainy.shape[0],1))
# testy = testy.reshape((testy.shape[0], 1))

# trainx = np.transpose(trainx)
# trainy = np.transpose(trainy)
# testx = np.transpose(testx)
# testy = np.transpose(testy)


# In[312]:


# print(trainx.shape)
# print(trainy.shape)
# print(testx.shape)
# print(testy.shape)


# In[287]:


# from keras.models import Sequential
# from keras.layers import Activation, Dense
# from keras import optimizers


# In[ ]:


# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=1320))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.fit(trainx, trainy, epochs=10, batch_size=32)


# In[294]:


# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=1320))
# model.add(Dense(10, activation='softmax'))
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# 生成虚拟数据


# In[318]:


# import numpy as np
# import keras

# # 将标签转换为分类的 one-hot 编码
# one_hot_labels = keras.utils.to_categorical(trainy, num_classes=3)


# In[119]:


# 训练模型，以 32 个样本为一个 batch 进行迭代
# model.fit(trainx, one_hot_labels, epochs=10, batch_size=32)


# In[124]:


dts = "'ARHGEF10L' 'PCBD1' 'RPS24' 'RPS27' 'RPS26' 'ERN1' 'ERN2' 'KBTBD11' 'COX8A' 'LOC440896' 'TP53AIP1' 'VARS2' 'LOC145820' 'SS18L1' 'MICAL2' 'C10orf2' 'RUSC1' 'OR6N2' 'C10orf4' 'RIMBP2' 'YIPF2' 'NAAA' 'SNORD123' 'NR3C2' 'TSC22D3' 'SFTA1P' 'TSC22D4' 'TSR1' 'SV2A' 'FCHSD1' 'FCHSD2' 'MTMR9L' 'CABC1' 'FCGR1A' 'LPP' 'DST' 'LOC572558' 'FIGF' 'GLP2R' 'CDC14B'"
dts  = dts.split()


# In[125]:


print(dts)


# In[132]:


rfs = "'ARHGEF10L' 'KBTBD11' 'LOC440896' 'OR6N2' 'TP53AIP1' 'VARS2' 'LOC145820' 'SS18L1' 'ERN2' 'MICAL2' 'RUSC1' 'C10orf4' 'SCGB2A1' 'MICAL1' 'CEACAM21' 'CEACAM20' 'RNY4' 'C10orf2' 'RNY5' 'ERN1' 'C14orf159' 'ITIH5' 'ALS2CL' 'SMYD1' 'GRIA4' 'COL10A1' 'PPP2R3A' 'DHRS7C' 'JUB' 'IL6R' 'VSTM2A' 'PAMR1' 'FIGF' 'ETV4' 'GLP2R' 'SPTBN1' 'CAV1' 'CCDC50' 'FOXN3' 'GPAM'"
rfs = rfs.split()


# In[131]:


ldas = "'RPS4Y1' 'DDX3Y' 'EIF1AY' 'SFTPB' 'UTY' 'AQP4' 'IGSF11' 'FOXE1' 'UPK1B' 'RGS20' 'GJB6' 'KRT4' 'SPOCK3' 'LHFPL3' 'ACSM2A' 'SERPINB3' 'ADH7' 'ALDH3A1' 'LIX1' 'FAM153A' 'VAV3' 'PRR15L' 'TJP3' 'SCGB2A2' 'SPDEF' 'PRSS8' 'AGR2' 'MGP' 'LUM' 'TFF1' 'EPCAM' 'C10orf81' 'KRT19' 'KRT18' 'RAB25' 'ADIPOQ' 'AZGP1' 'GATA3' 'AGR3' 'KIAA1324'"
ldas = ldas.split()


# In[140]:


svms = "'HBA1' 'SERPINA6' 'SCGB1A1' 'KRT4' 'CES1' 'HBB' 'HBA2' 'KRT13' 'KCNC2' 'CES4' 'LOC162632' 'AKR1B10' 'KIAA0408' 'CHGB' 'C4orf7' 'C14orf180' 'BMP5' 'ATP1A2' 'NRG3' 'UGT2B28' 'SLC30A8' 'CYP4Z1' 'LOC283867' 'CCL17' 'PITX1' 'TNNT1' 'VSTM2L' 'HSD17B6' 'CSN1S2A' 'MMP11' 'TMEM92' 'GSTM1'  'WT1' 'COMP' 'FAM5C' 'CST1' 'ALOX15' 'PPAPDC1A' 'MS4A15' 'COL10A1'"
svms = svms.split()


# In[ ]:


a = []
a.append(dts)
a.append(rfs)
a.append(ldas)
a.append(svms)


# In[162]:


temp ={'DT':dts,'RFS':rfs,
       'LDA':ldas,'SVM':svms}


# In[163]:


b=pd.DataFrame(temp)


# In[164]:


df  = b.apply(pd.value_counts)


# In[ ]:


print(b.apply(pd.value_counts))


# In[165]:


df['Col_sum'] = df.apply(lambda x: x.sum(), axis=1)


# In[167]:


print(df.sort_values(by="Col_sum" , ascending=False))


# In[ ]:


from collections import Counter


# In[169]:


print(df.index)


# In[173]:




def get_index_list(data):
    p = data

    index_list = []
    for i in data:
        index_list.append(i[1:-1])
    print((index_list))
    return index_list
features1st = get_index_list(df.index)


# In[179]:


b = pd.DataFrame(result[features1st])


# In[215]:


print(b)


# In[214]:


newsick = b.index.values 
newsick = newsick.tolist()
print (newsick.count(0)) 
print (newsick.count(1)) 
print (newsick.count(2))


# In[ ]:


ntrain_x = [b.iloc[j] for j in train_index]
ntrain_y = [newsick[j] for j in train_index]
ntest_x = [b.iloc[j] for j in range(0, 1652) if j not in train_index]
ntest_y = [newsick[j] for j in range(0, len(sick)) if j not in train_index]


# In[217]:





# In[241]:


from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier()
tree_clf.fit(ntrain_x, ntrain_y)


# In[243]:


nprobas = tree_clf.predict(ntest_x)
precision_recall_fscore_support(ntest_y, probas,average='macro')

mean_squared_error(ntest_y, nprobas)


# In[ ]:


random_forest_reg = ensemble.RandomForestRegressor()
random_forest_reg.fit(ntrain_x, ntrain_y)

##plot the ROC curve for the training set and compute the auc_score
training_set_z_rf = random_forest_reg.predict(ntest_x)
nprobas = random_forest_reg.predict(ntest_x)


# In[202]:


from sklearn.metrics import mean_squared_error


# In[203]:


mean_squared_error(ntest_y, nprobas)


# In[240]:


mean_squared_error(test_y, probas)


# In[230]:



random_state = np.random.RandomState(0)
new_svm =  svm.SVC(kernel = 'linear', probability = True, )
new_svm.fit(train_x, train_y)
probas = new_svm.predict(test_x)
precision_recall_fscore_support(test_y, probas,average='macro')


# In[231]:


mean_squared_error(test_y, probas)


# In[234]:


from sklearn.svm import SVC
random_state = np.random.RandomState(1)
nsvm = SVC(gamma='auto')
nsvm.fit(ntrain_x, ntrain_y)

nprobas = nsvm.predict(ntest_x)


# In[235]:


precision_recall_fscore_support(ntest_y, nprobas,average='macro')


# In[236]:


mean_squared_error(ntest_y, nprobas)


# In[225]:





# In[229]:


mean_squared_error(ntest_y, nprobas)


# In[223]:


print(ntest_x)


# In[ ]:




