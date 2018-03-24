from os import listdir
import gensim
from gensim.models import Doc2Vec
import gensim.models.doc2vec
import re
import math
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
import numpy as np
from sklearn.datasets import load_svmlight_file,dump_svmlight_file
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm

x =[1]* 12500
y = [0] * 12500
y_train = np.concatenate((x,y),axis=0)
y_test = y_train 

model =gensim.models.doc2vec.Doc2Vec.load("F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\doc2vec_model")

def method_Log_regression(x_train,x_test):
    clf = linear_model.SGDClassifier(loss='log')
    clf.fit(x_train, y_train)
    p=clf.predict(x_test)
    accu = roc_auc_score(y_test,p)
    accu = accu*100
    print("Accuracy on Paragraph vector with Logistic regression :" "%.4f" % accu)
    
    

def method_feed_forward(x_train,x_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,8), random_state=1)
    clf.fit(x_train, y_train) 
    p=clf.predict(x_test)
    accu = roc_auc_score(y_test,p)
    accu = accu*100
    print("Accuracy on Paragraph vector with feed forward Neural : " "%.4f" % accu)
    
    
def method_SVM(x_train,x_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train) 
    p=clf.predict(x_test)
    accu = roc_auc_score(y_test,p)
    accu = accu*100
    print("Accuracy on Paragraph vector with SVM calssifier : " "%.4f" % accu)


def process_docs(directory):
    data = []
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue
        # create the full path of the file to open
        path = directory + '\\' + filename
        docvec = model.infer_vector(path)
        data.append(docvec)
    return (data)

def test_data(directory):
    #prepare negative reviews
    neg_data = process_docs(directory + '\\neg')
    # prepare positive reviews
    pos_data= process_docs(directory + '\\pos')
    for i in range(0,len(neg_data)):
        pos_data.append(neg_data[i])
    return (pos_data)
def train_data(directory):
    data=[]
    path =directory + '\\pos'
    for filename in listdir(path):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue
        docvec = model[filename]
        data.append(docvec)
    path =directory + '\\neg'
    for filename in listdir(path):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue
        docvec = model[filename]
        data.append(docvec)
    return data

def method(direc1 , direc2):
    x_train = train_data(direc1)
    x_test = test_data(direc2)
    method_Log_regression(x_train,x_test)
    method_feed_forward(x_train,x_test)
    method_SVM(x_train,x_test)

testing_directory=('F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\aclImdb\\test') 
training_directory=('F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\aclImdb\\train')

method(training_directory,testing_directory)