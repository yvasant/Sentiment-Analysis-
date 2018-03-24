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
from scipy.sparse import find
from sklearn.neural_network import MLPClassifier

model = Word2Vec.load("F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\word2vec_vectors")
WWords = list(model.wv.vocab)
WWords = set(WWords)
my_vocab = (open('F:/ACADS/acads 6th sem/NLP/Assignments/Assign 2/aclImdb/imdb.vocab',encoding="utf8").read().split())
tfidf_train = load_svmlight_file("F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\train_TFIDF.feat")
tfidf_test = load_svmlight_file("F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\test_TFIDF.feat")
x =[1]* 12500
y = [0] * 12500
y_train = np.concatenate((x,y),axis=0)
y_test = y_train
X_train, Y_train = load_svmlight_file("F:/ACADS/acads 6th sem/NLP/Assignments/Assign 2/aclImdb/train/labeledBow.feat")
X_test , Y_test =  load_svmlight_file("F:/ACADS/acads 6th sem/NLP/Assignments/Assign 2/aclImdb/test/labeledBow.feat")
def method_Log_regression(x_train,x_test):
    clf = linear_model.SGDClassifier(loss='log')
    clf.fit(x_train, y_train)
    p=clf.predict(x_test)
    accu = roc_auc_score(y_test,p)
    accu = accu*100
    print("Accuracy on Avg_Word2vec with TFIDF and Logistic regression : " ,"%.4f" % accu)
    
      
def method_feed_forward(x_train,x_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,8), random_state=1)
    clf.fit(x_train, y_train) 
    p=clf.predict(x_test)
    accu = roc_auc_score(y_test,p) 
    accu = accu*100
    print("Accuracy on Avg_Word2vec with TFIDF and feed forward Neural : " "%.4f" % accu)
    
    
def method_SVM(x_train,x_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train) 
    p=clf.predict(x_test)
    accu = roc_auc_score(y_test,p)
    accu = accu*100
    print("Accuracy on Avg_Word2vec with TFIDF and SVM calssifier : " "%.4f" % accu)
    

def process_docs1(tfidf,X):
    vec = []
    a = tfidf.nonzero()
    
    for i in range(25000):
        rows, cols, value = find(tfidf[i])
        vect = np.sum([model[my_vocab[j]] for j in cols if my_vocab[j] in WWords], axis=0 )/np.sum(X[i])
        vec.append(vect)
                
    return vec

def method():
    x_train = process_docs1(tfidf_train[0],X_train)
    x_test = process_docs1(tfidf_test[0],X_test)
    
    method_Log_regression(x_train,x_test)
    method_feed_forward(x_train,x_test)
    method_SVM(x_train,x_test)

#call to methods 

method()
