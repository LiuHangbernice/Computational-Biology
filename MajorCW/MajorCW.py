#!/usr/bin/env python
# coding: utf-8

# In[99]:


# coding: utf-8
log = print

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


# In[ ]:


##load the data to pandas dataframe
data = pd.read_csv('tcga_RSEM_Hugo_norm_count.gz', compression='gzip', delim_whitespace = True)
#data = pd.read_csv('HiSeqV2_PANCAN.gz', compression='gzip', delim_whitespace = True)

print (data.info())


# In[142]:


print(data.T)


# In[135]:


data2 = pd.read_csv('HiSeqV2_PANCAN_Z.gz', compression='gzip', delim_whitespace = True)


# In[143]:


print(data2.T)


# In[144]:


print(type(data2))
print(type((data2.T)))


# In[147]:


data_clean = data.select_dtypes(include=['float64'])
column_names = list(data_clean.columns.values)


# In[4]:





# In[5]:


print(data_clean)

print(len(data_clean))


# In[7]:


label = [s[-2:] for s in column_names]



label = [0 if s in ('01', '02', '03', '04', '05', '06', '07', '08', '09') else 1 for s in label]


# In[49]:


train_index = random.sample(range(0, len(label)), 1000)
train_x = [data_clean.iloc[:,j] for j in train_index]
train_y = [label[j] for j in train_index]
test_x = [data_clean.iloc[:,j] for j in range(0, len(label)) if j not in train_index]
test_y = [label[j] for j in range(0, len(label)) if j not in train_index]


# In[ ]:


from sklearn.decomposition import PCA
X = train_x
pca = PCA(n_components=2)
pca.fit(X)


# In[ ]:


print(len(data['sample']))

# for i in range(len(data['sample'])):

#     if data['sample'][i] == 'FN1':
#         print(i)
np.corrcoef(p,label)[0][1]


# In[ ]:


colist = []

for i in range(58581):
    #print(data_clean.iloc[30559])
    p = data_clean.iloc[i]
    temp = np.corrcoef(p,label)[0][1] 
    colist.append(temp)
    
#     if np.corrcoef(p,label)[0][1] > 0.9 or np.corrcoef(p,label)[0][1] < -0.9:
#         print(i)
#         print(np.corrcoef(p,label))
        
print('end')


# In[128]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()


clf.fit(train_x, train_y)


# In[132]:


#print(clf.scalings_ )
plot_coefficients(clf, data['sample'])


# In[68]:


new_x  = clf.transform(test_x)

print(test_x[0])


log(clf.score(test_x, test_y))


# In[66]:


print(new_x[0])


# In[ ]:


max(colist)

temp = []
temp2 = []
for i in range(len(colist)):
    if colist[i] > 0.35:
        temp.append(i)
    if colist[i] < -0.38:
         temp2.append(i)

print(len(temp))
print(len(temp2))


# In[ ]:


## create a log file


# Set the logger
# . set_logger(os.path.join('', 'train_various_layers.log'))


### Feature Reduction with prior knowledge

##load the cell cycle gene set into a numpy array
select_genes = np.loadtxt('cell_cycle geneset.txt', dtype = np.dtype('str'), skiprows=2) 



print(select_genes.shape)
#logging.info("cell_cycle geneset")


# In[ ]:


##load the cell death gene set into a numpy array
select_genes = np.loadtxt('cell_death geneset.txt', dtype = np.dtype('str'), skiprows=2) 
print(select_genes.shape)
#logging.info("cell_death geneset")


##load the cell adhesion gene set into a numpy array
select_genes = np.loadtxt('cell_adhesion geneset.txt', dtype = np.dtype('str'), skiprows=2) 
print(select_genes.shape)
#logging.info("cell_adhesion geneset")




##combine the 3 gene sets
select_genes = np.loadtxt('combined geneset.txt', dtype = np.dtype('str'), skiprows=2)
print(select_genes.shape)
#logging.info("combined geneset")




##select only genes in the cell cycle pathway
data_selected = data.loc[data['sample'].isin(select_genes)]
print (data_selected.shape)
#logging.info("cell cycle gene set selected data shape: " + str(data_selected.shape))


# In[ ]:


##clean the data
data_clean = data.select_dtypes(include=['float64'])
column_names = list(data_clean.columns.values)
label = [s[-2:] for s in column_names]
print(len(label))
#label = [0 if s in ('01', '02', '03', '04', '05', '06', '07', '08', '09') else 1 for s in label]

print (label.count(0)) #cancer (9807)
print (label.count(1)) #non-cancer (856)


# In[ ]:


##split the data randomly into training and test
train_index = random.sample(range(0, len(label)), 9000)
train_x = [data_clean.iloc[:,j] for j in train_index]
train_y = [label[j] for j in train_index]
test_x = [data_clean.iloc[:,j] for j in range(0, len(label)) if j not in train_index]
test_y = [label[j] for j in range(0, len(label)) if j not in train_index]
print(len(train_index))
print(test_x)
print(test_y)


# In[ ]:


### Model 1: Logistic regression with k-fold cross-validation and roc curves
X = data_clean.T
y = np.array(label)

random_state = np.random.RandomState(0)

cv = StratifiedKFold(n_splits = 2)
logreg = linear_model.LogisticRegression(penalty = 'l1', C=1e5, 
                                        random_state = random_state)

f1_scores = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas = logreg.fit(X.iloc[train], y[train]).predict(X.iloc[test])
    print(X.iloc[train].shape, y[train].shape, probas.shape)
    fpr, tpr, thresholds = roc_curve(y[test], probas)
    f1_score = metrics.f1_score(y[test], probas)
    tprs.append(interp(mean_fpr, fpr, tpr))
    f1_scores.append(f1_score)
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw = 1, alpha = 0.3, label = "ROC fold %d (AUC = %0.2f)" % (i, roc_auc))
    i += 1


# In[ ]:


plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', label = 'Luck', alpha = 0.8)
mean_tpr = np.mean(tprs, axis = 0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#plot F1 scores
average = sum(f1_scores)/len(f1_scores)
plt.plot(f1_scores)
plt.ylim(0.0, 1.0)
plt.xlabel('fold')
plt.ylabel('f1_score')
plt.title('F1_score across 7_fold')
plt.axhline(y = average, color = 'r')
plt.show()


# In[ ]:


### Model 2: Random forest

print(test_x)


# In[ ]:


random_forest_reg = ensemble.RandomForestRegressor()
random_forest_reg.fit(train_x, train_y)

##plot the ROC curve for the training set and compute the auc_score
training_set_z_rf = random_forest_reg.predict(test_x)

fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(test_y, training_set_z_rf)
training_auc_score_rf = metrics.roc_auc_score(test_y, training_set_z_rf)
plt.plot(fpr_rf, tpr_rf)
plt.title("random_forest_test_roc")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.show()
print("Random Forest Training AUC: " + str(training_auc_score_rf))


# In[ ]:





# In[ ]:


X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])


# In[103]:


log(random_forest_reg.feature_importances_.ravel())


# In[ ]:


### Model 3: Support vector machine (SVM)
X = data_clean.T[1:1000]
y = np.array(label)[1:1000]


# In[133]:


train_x = data_clean.T[1:1000]
train_y = np.array(label)[1:1000]
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
    


# In[131]:


# svm =  svm.SVC(kernel = 'linear', probability = True, 
#                     random_state = random_state)
# svm.fit(train_x, train_y)
plot_coefficients(svm, data['sample'])


# In[ ]:


random_state = np.random.RandomState(0)
svm =  svm.SVC(kernel = 'linear', probability = True, )
svm.fit(train_x, train_y)
plot_coefficients(svm, data['sample'])


# In[96]:





# In[ ]:


random_state = np.random.RandomState(0)
cv = StratifiedKFold(n_splits = 7)
classifier = svm.SVC(kernel = 'linear', probability = True, 
                    random_state = random_state)

f1_scores = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0


# In[ ]:


print(X.shape)
print(y.shape)

a = np.array([1, 1, 2, 2])
b = np.array([0.1, 0.4, 0.35, 0.8])
print(a.shape)
print(b.shape)


# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


from sklearn.metrics import precision_recall_fscore_support

for train, test in cv.split(X, y):
    probas = classifier.fit(X.iloc[train], y[train]).predict(X.iloc[test])
    #fpr, tpr, thresholds = roc_curve(y[test], probas)

#     print(type(y[test]))
#     print(y[test])
    f1_score =  precision_recall_fscore_support(y[test], probas,average='macro')
    print(list(f1_score))
# #     tprs.append(interp(mean_fpr, fpr, tpr))
    #f1_scores.append(f1_score)
#     tprs[-1][0] = 0.0
#     roc_auc = auc(fpr, tpr)
#     aucs.append(roc_auc)
#     plt.plot(fpr, tpr, lw = 1, alpha = 0.3, label = "ROC fold %d (AUC = %0.2f)" % (i, roc_auc))

    i += 1
    


# In[ ]:


from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []

for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


plt.show()


# In[ ]:


# plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', label = 'Luck', alpha = 0.8)
# mean_tpr = np.mean(tprs, axis = 0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# plt.plot(mean_fpr, mean_tpr, color='b',
#          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#          lw=2, alpha=.8)

# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                  label=r'$\pm$ 1 std. dev.')

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

#plot F1 scores
average = sum(f1_scores)/len(f1_scores)
plt.plot(f1_scores)
plt.ylim(0.0, 1.0)
plt.xlabel('fold')
plt.ylabel('f1_score')
plt.title('F1_score across 7_fold')
plt.axhline(y = average, color = 'r')
plt.show()


# In[81]:


from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier()
tree_clf.fit(train_x, train_y)


# In[119]:


plot_coefficients(tree_clf, data['sample'])


# In[82]:


#data['sample']


# In[83]:


from sklearn.tree import export_graphviz
export_graphviz(tree_clf,
                out_file="tree.dot",
                feature_names=data['sample'],
                rounded=True,
                filled=True
               )


# In[84]:


#!apt install graphviz 

# Convert dot file to png file.
get_ipython().system('dot -Tpng tree.dot -o tree.png')


# In[85]:


from IPython.display import Image
Image(filename='tree.png')


# In[ ]:





# In[156]:


### Model 4: Neural network with TensorFlow 
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import optimizers
## turn the data into the right shape
train_x = [data_clean.iloc[:,j] for j in train_index]
train_y = [label[j] for j in train_index]


# In[157]:


train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
print(train_x)
print(train_y)


# In[158]:


train_y = train_y.reshape((train_y.shape[0],1))

test_y = test_y.reshape((test_y.shape[0],1))


# In[ ]:





# In[ ]:


# data1 = np.random.random((1000, 100))
# labels1 = np.random.randint(2, size=(1000, 1))
test_y = test_y.reshape((test_y.shape[0],1))

score = model.evaluate(test_x, test_y, batch_size=128)


# In[ ]:





# In[149]:



train_index = random.sample(range(0, len(label)), 1000)
train_x = [data_clean.iloc[:,j] for j in train_index]
train_y = [label[j] for j in train_index]
test_x = [data_clean.iloc[:,j] for j in range(0, len(label)) if j not in train_index]
test_y = [label[j] for j in range(0, len(label)) if j not in train_index]


# In[ ]:


trainx = np.array(train_x)
trainy = np.array(train_y)
testx = np.array(test_x)
testy = np.array(test_y)

trainy = trainy.reshape((trainy.shape[0],1))
testy = testy.reshape((testy.shape[0], 1))

trainx = np.transpose(trainx)
trainy = np.transpose(trainy)
testx = np.transpose(testx)
testy = np.transpose(testy)
print(trainx.shape)


# In[153]:


print(trainy.shape)


# In[154]:





##create placeholders
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an feature vector (58,581 genes)
    n_y -- scalar, number of classes
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    """

    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])
    
    return X, Y



# run placeholder function

X, Y = create_placeholders(58581, 1)
print ("X = " + str(X))
print ("Y = " + str(Y))




##initialize_parameters
def initialize_parameters(dimensions = [58581, 10, 10, 10, 10, 10, 10, 10, 1]):
    """
    dimensions = [n_x, number of neurons in each layer, n_y]
    Initializes parameters to build a neural network with tensorflow.                       
    
    Returns:
    parameters -- a dictionary of tensors containing Ws and bs
    """
    
    tf.set_random_seed(1)                   # so that our results are reproducible
    parameters = {}
    for i in range(len(dimensions) - 1):
        parameters["W" + str(i+1)] = tf.get_variable("W" + str(i+1), [dimensions[i+1], dimensions[i]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters["b" + str(i+1)] = tf.get_variable("b" + str(i+1), [dimensions[i+1], 1], initializer = tf.zeros_initializer())
    
    logging.info("number of layers: " + str(len(dimensions)-1))
    for j in range (len(dimensions)-1):
        logging.info("number of neurons in " + str(j+1) + " layer: " + str(dimensions[j+1]))

    return parameters


# run initialize_parameters function
tf.reset_default_graph()
# with tf.Session() as sess:
#     parameters = initialize_parameters([58581, 1000, 500, 500, 250, 1])




##forward propagation
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    A = X
    
    # Retrieve the parameters from the dictionary "parameters" 
    for i in range (int(len(parameters.keys()) / 2)):
        W = parameters["W" + str(i+1)]
        b = parameters["b" + str(i+1)]
        Z = tf.add(tf.matmul(W, A), b)
        A = tf.nn.relu(Z)

    return Z



# run forward_propagation

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(58581, 1)
    parameters = initialize_parameters([58581, 10, 10, 10, 10, 10, 10, 10, 1])
    Z = forward_propagation(X, parameters)
    print("Z = " + str(Z))




##compute cost

def compute_cost(Z, Y, pos_weight = 30):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    targets -- vector of labels y (1 or 0) 
    
    
    Returns:
    cost -- runs the session of the cost 
    """
    logging.info("pos_weight: " + str(pos_weight))
    
    # to fit the tensorflow requirement for tf.nn.weighted_cross_entropy_with_logits(...,...)
    logits = Z
    targets = Y
    
    # Use the loss function, pos_weight is set randomly, will tune later
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets = targets, logits = logits, pos_weight = pos_weight))
    
    return cost



# run compute_cost

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(58581, 1)
    parameters = initialize_parameters([58581, 10, 10, 10, 10, 10, 10, 10, 1])
    Z = forward_propagation(X, parameters)
    cost = compute_cost(Y, Z)
    print("cost = " + str(cost))




# final model
def model(X_train, Y_train, X_test, Y_test, dimensions = [1000, 500, 500, 250], 
          learning_rate = 0.0001, 
          num_epochs = 100, pos_weight = 30, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X_train -- training set, of shape (input size = 20530, number of training examples = 9000)
    Y_train -- test set, of shape (output size = 1, number of training examples = 9000)
    X_test -- training set, of shape (input size = 20530, number of training examples = 1663)
    Y_test -- test set, of shape (output size = 1, number of test examples = 1663)
    learning_rate -- learning rate of the optimization
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    logging.info("learning_rate: " + str(learning_rate))
    logging.info("num_epochs: " + str(num_epochs))
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    

    # Initialize parameters
    parameters = initialize_parameters(dimensions)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z = forward_propagation(X, parameters)
    
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z, Y, pos_weight)
    
    
    # L2 regularization
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1, scope=None)
    weights = tf.trainable_variables() # all vars of your graph
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
    regularized_loss = cost + regularization_penalty # this loss needs to be minimized

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(regularized_loss)
    
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            #epoch_cost = 0.                       # Defines a cost related to an epoch
        
            
            # IMPORTANT: The line that runs the graph on a minibatch.
            # Run the session to execute the "optimizer" and the "cost"
            _ , batch_cost = sess.run((optimizer, cost), feed_dict = {X: X_train, Y: Y_train})

                
            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, batch_cost))
            if print_cost == True and epoch % 10 == 0:
                costs.append(batch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()


        # save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        sigmoid_Z = tf.sigmoid(Z) 
        predictions = tf.cast(tf.map_fn(lambda x: tf.round(x), sigmoid_Z), tf.int32)
        
        
        
        # Precision and recall For training set
        

        Y_train_placeholder = tf.placeholder(tf.int32, shape = Y_train.shape)
        train_predictions_placeholder = tf.placeholder(tf.int32, shape=Y_train.shape)
        train_precision, train_precision_update_op = tf.metrics.precision(labels = Y_train_placeholder, predictions = train_predictions_placeholder)
        
        train_recall, train_recall_update_op = tf.metrics.recall(labels = Y_train_placeholder, predictions = train_predictions_placeholder)
        sess.run(tf.local_variables_initializer())
        train_predictions = sess.run(predictions, feed_dict = {X: X_train})

        train_precision_value, train_recall_value = sess.run([train_precision_update_op, train_recall_update_op], feed_dict={Y_train_placeholder: Y_train, train_predictions_placeholder: train_predictions})

        
        # Precision, recall for test set
        Y_test_placeholder = tf.placeholder(tf.int32, shape=Y_test.shape) 
        predictions_placeholder = tf.placeholder(tf.int32, shape=Y_test.shape)
        precision, precision_update_op = tf.metrics.precision(labels = Y_test_placeholder, predictions = predictions_placeholder)
        
        recall, recall_update_op = tf.metrics.recall(labels = Y_test_placeholder, predictions = predictions_placeholder)
        sess.run(tf.local_variables_initializer())
        final_predictions = sess.run(predictions, feed_dict={X: X_test})
#         print(final_predictions)
#         print(np.squeeze(Y_test))

        precision_value, recall_value = sess.run([precision_update_op, recall_update_op], feed_dict={Y_test_placeholder: Y_test, predictions_placeholder: final_predictions})

        train_F1_score = 2 * train_precision_value * train_recall_value / (train_precision_value + train_recall_value)
        logging.info("train_precision_value: " + str(train_precision_value))
        logging.info("train_recall_value: " + str(train_recall_value))
        logging.info("train_F1_score: " + str(train_F1_score))
        
        F1_score = 2 * precision_value * recall_value / (precision_value + recall_value)
        logging.info("final_cost: " + str(costs[-1]))
        logging.info("precision_value: " + str(precision_value))
        logging.info("recall_value: " + str(recall_value))
        logging.info("F1_score: " + str(F1_score))
        

        return parameters


# In[ ]:





# In[ ]:


### Integrated Gradients

from deepexplain import DeepExplain

X = tf.placeholder(tf.float32, shape=[58581, None])
tf.reset_default_graph()

# Create a DeepExplain context
with DeepExplain(session = sess) as de:
    parameters = model(train_x, train_y, test_x, test_y, 1000, 500, 0.00001, 100, 16)
    
    def raw_model(x, act=tf.nn.relu):  # < different activation functions lead to different explanations
        layer_1 = act(tf.add(tf.matmul(parameters[0]['W1'], x), parameters[0]['b1']))
        layer_2 = act(tf.add(tf.matmul(parameters[0]['W2'], layer_1), parameters[0]['b2']))
        out_layer = tf.matmul(parameters[0]['W3'], layer_2) + parameters[0]['b3']
        return out_layer

    # Construct model
    logits = raw_model(X)
    
    print(logits)
    
    attributions = {
        'Integrated Gradients': de.explain('intgrad', logits * test_y, X, test_x)
    }
    print('Done')
    print(attributions['Integrated Gradients'])



## obtained the index of the most important genes
ig = attributions['Integrated Gradients']
rows, cols = np.where(ig > 0.065)
print(len(set(rows)))  ##5136 IDs with attribution scores greater than 0.05
gene_index = set(rows)
most_relevant_genes = data['sample'][gene_index]
print(most_relevant_genes)
np.savetxt('most_relevant_genes.txt', most_relevant_genes, fmt = '%s')


# In[ ]:





# In[124]:


a = ['COL10A1', 'CSN1S2A', 'PPAPDC1A' ,'CYP4F8', 'CSN2', 'CST2', 'MMP11' ,'COMP',
 'TMEM90B' ,'SLC24A2', 'CYP4Z1', 'LRRC15', 'TMEM92' ,'KLK4' ,'CEL' ,'CELP',
 'CTHRC1', 'FNDC1', 'GC', 'MATN3', 'TSPAN8' ,'GJB6' ,'TMPRSS4' ,'MYOC' ,'ENPP3',
 'BCHE' ,'DLK1' ,'ADH4', 'GRIN2A' ,'PPAN-P2RY11' ,'UGT2B28', 'MASP1', 'HBA2',
 'KRT4' ,'HBB', 'CES1' ,'HBA1' ,'AKR1B10' 'PENK' ,'KRT13']

b = ['ARHGEF10L' ,'PCBD1', 'RPS24' ,'RPS27', 'RPS26' ,'ERN1' ,'ERN2' ,'KBTBD11',
 'COX8A' ,'LOC440896', 'TP53AIP1' ,'VARS2' ,'LOC145820', 'SS18L1' ,'MICAL2',
 'C10orf2', 'RUSC1', 'OR6N2' ,'C10orf4', 'RIMBP2', 'MC1R' ,'LOC728264' ,'ATRX',
 'GNA11', 'PPAPDC1A', 'TRAK1', 'C4orf38' ,'KIF26B' ,'SPC25' ,'PAMR1', 'PPP1R12B',
 'TMEM39A' ,'RASL11B','NPNT' ,'ARHGAP20', 'CREB3L1' ,'CD300LG' ,'SPRY2',
 'ADAMTS5' ,'FIGF']

c = ['ARHGEF10L', 'PCBD1', 'RPS24' ,'RPS27', 'RPS26', 'ERN1', 'ERN2' ,'KBTBD11',
 'COX8A', 'LOC440896', 'TP53AIP1', 'VARS2','LOC145820', 'SS18L1', 'MICAL2',
 'C10orf2' ,'RUSC1' ,'OR6N2', 'C10orf4' ,'RIMBP2' ,'SV2A' ,'MTMR9L', 'CABC1',
 'FCHSD2' ,'FCHSD1', 'MTMR1' ,'SELS', 'TSR1' ,'TSC22D4' ,'SFTA1P', 'TSC22D3',
 'NR3C2' ,'NAAA', 'OR11A1' ,'CHST6', 'KANK2', 'UHRF1', 'FIGF', 'COL11A1',
 'ADAMTS5']

d = ['SNAR-C2', 'SNORA35', 'SNAR-H' ,'SNORD12B', 'SNAR-G2', 'SNORD88B' ,'DEFB116',
 'SNAR-D' ,'SNORD16' ,'SNORD88C', 'TTTY17B' ,'SNORA11C', 'SNORD46' ,'TTTY22',
 'DEFB114', 'SNORD124' ,'PRR20B', 'TTTY11', 'SNORA70C', 'TSSK2' 'TTTY7',
 'KRTAP23-1', 'S100A7L2', 'TXNDC8', 'DEFB112', 'KRTAP19-7','DEFB121',
 'LOC653545', 'PRAMEF3', 'TTTY3B' ,'?|136542', 'DUX4', 'SNORD11B' ,'KRTAP20-3',
 'KRTAP22-1', 'VTRNA1-1', 'PRR20D', 'SNORD90', 'TTTY17A', 'SNORD115-20']

for i in a:
    if i in b:
        print(i)
    if i in c:
        print(i)
    if i in d:
        print(i)


# In[127]:


for i in c:
    if i in b:
        print(i)
    if i in d:
        print(i)


# In[ ]:




