# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 01:39:52 2018

@author: vasant
"""
from sklearn.datasets import load_svmlight_file
from os import listdir 
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
import gensim
from gensim.models import Word2Vec
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier

x =[1]* 12500
y = [0] * 12500
y_train = np.concatenate((x,y),axis=0)
y_test = y_train

def method_Log_regression(x_train,x_test):
    clf = linear_model.SGDClassifier(loss='log')
    clf.fit(x_train, y_train)
    p=clf.predict(x_test)
    accu = roc_auc_score(y_test,p)
    accu = accu*100
    print("Accuracy on TF with Logistic regression :" "%.4f" % accu)
    
    
def method_Naive_bayes(x_train,x_test):
    model = BernoulliNB().fit(x_train, y_train)
    p = model.predict(x_test)
    accu = roc_auc_score(y_test,p)
    accu = accu*100
    print("Accuracy on TF with Naive bayes calssifier : " ,"%.4f" % accu)
    
def method_feed_forward(x_train,x_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,8), random_state=1)
    clf.fit(x_train, y_train) 
    p=clf.predict(x_test)
    accu = roc_auc_score(y_test,p)
    accu = accu*100
    print("Accuracy on TF with feed forward Neural : " "%.4f" % accu)
    
    
def method_SVM(x_train,x_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train) 
    p=clf.predict(x_test)
    accu = roc_auc_score(y_test,p)
    accu = accu*100
    print("Accuracy on TF with SVM calssifier : " "%.4f" % accu)
    
def method(direc1, direc2):
    x_train = load_svmlight_file(direc1)
    x_test = load_svmlight_file(direc2)
    method_Log_regression(x_train[0],x_test[0])
    method_Naive_bayes(x_train[0],x_test[0])
    method_feed_forward(x_train[0],x_test[0])
    method_SVM(x_train[0],x_test[0])

#call to methods 

training_direc = 'F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\train_TF.feat'
testing_direc = 'F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\test_TF.feat'
method(training_direc,testing_direc)

