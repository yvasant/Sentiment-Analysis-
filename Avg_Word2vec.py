from os import listdir
import nltk 
import re
import math
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
import gensim
from gensim.models import Word2Vec
import numpy as np
from sklearn.datasets import load_svmlight_file,dump_svmlight_file
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier

x =[1]* 12500
y = [0] * 12500
y_train = np.concatenate((x,y),axis=0)
y_test = y_train 
TAG_RE = re.compile(r'<[^>]+>')

def method_Log_regression(x_train,x_test):
    clf = linear_model.SGDClassifier(loss='log')
    clf.fit(x_train, y_train)
    p=clf.predict(x_test)
    accu = roc_auc_score(y_test,p)
    accu = accu*100
    print("Accuracy on Avg Word2vec with Logistic regression :" "%.4f" % accu)
    
    
def method_feed_forward(x_train,x_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,8), random_state=1)
    clf.fit(x_train, y_train) 
    p=clf.predict(x_test)
    accu = roc_auc_score(y_test,p)
    accu = accu*100
    print("Accuracy on Avg Word2vec with feed forward Neural : " "%.4f" % accu)
    
    
def method_SVM(x_train,x_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train) 
    p=clf.predict(x_test)
    accu = roc_auc_score(y_test,p)
    accu = accu*100
    print("Accuracy on Avg Word2vec with SVM calssifier : " "%.4f" % accu)
    

def remove_tags(text):
    return TAG_RE.sub('', text)

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding="utf-8")
    # read all text
    #print(filename)
    text = file.read()
    file.close()
    text = remove_tags(text)
    return text

model = Word2Vec.load("F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\word2vec_vectors")
WWords = list(model.wv.vocab)
WWords = set(WWords)


def process_docs(directory):
    neg_vec = []
    for filename in listdir(directory):
        #print(filename)
        path = directory + '\\' + filename
        doc = load_doc(path)
        words = word_tokenize(doc)
        avg_vec = np.sum([model[word] for word in words if word in WWords],axis=0)/(len(words))
        neg_vec.append(avg_vec)
    return neg_vec

def training(directory):
    neg_vecs = process_docs(directory + '\\neg')
    pos_vecs = process_docs(directory + '\\pos')
    return np.concatenate((pos_vecs,neg_vecs),axis=0)

def testing(directory):
    neg_vecs = process_docs(directory + '\\neg')
    pos_vecs = process_docs(directory + '\\pos')
    return np.concatenate((pos_vecs,neg_vecs),axis=0)

def method(direc1 , direc2):
    x_train = training(direc1)
    x_test = testing(direc2)
    method_Log_regression(x_train,x_test)
    method_feed_forward(x_train,x_test)
    method_SVM(x_train,x_test)

#call to methods

training_direc = 'F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\aclImdb\\train'
testing_direc = 'F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\aclImdb\\test'
method(training_direc,testing_direc)

