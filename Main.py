# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 01:39:52 2018

@author: vasant
"""

from sklearn.datasets import load_svmlight_file,dump_svmlight_file
import re
import math 
from os import listdir
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cross_validation import train_test_split
import gensim
from gensim.models import Word2Vec
import gensim.models.doc2vec
import numpy as np


def get_data(directory):
        f=open(directory,'r').read()
        data = load_svmlight_file(directory)
        lines = [line.rstrip('\n') for line in open(directory, 'r')]
        return data,lines,f
p = re.compile("([\:][0-9]*)")
q =re.compile("[\s][0-9]*[:]")

#training and testing lables
x =[1]* 12500
y = [0] * 12500
y_train = np.concatenate((x,y),axis=0)
y_test = y_train 
# Binary Bag of Word is implemented here 

def BBOW(p, data ,lines,st):
    quote=""
    finalfile = open(st + '.txt' , 'w')
    for i in range(data[0].shape[0]):
        quote = (p.sub(r':1', lines[i]))
        finalfile.write(quote)
        finalfile.write('\n')
    finalfile.close()


# Term frequency is implemented here

def tf(p , lines ,data ,st):
    sum1 = 0
    finalfile = open(st + '_TF.txt' , 'w')
    tftext = ""
    for i in range(len(lines)):
        iterator = p.finditer(lines[i])
        for match in iterator:
            x=match.span()[0]+1
            y = match.span()[1]-1
            if x==y:
                num = int(lines[i][x])
            else:
                num = int(lines[i][x:y])
            sum1 = sum1 + num
        f = str(round((1/sum1),10))
        tftext = p.sub(r''+':'+ f,lines[i])
        finalfile.write(tftext)
        finalfile.write('\n')
        sum1 = 0
    finalfile.close()
    tfmult = load_svmlight_file("F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\"+st +"_TF.txt")
    finalTF = data[0].multiply(tfmult[0])
    dump_svmlight_file(finalTF,data[1], st + "_TF.feat")

# Term frequency - inverse document frequency is implemented here

def tfidf(p,st,f):
    #p==q
    tf_data = load_svmlight_file("F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\" + st +"_TF.feat")
    iterator = p.finditer(f)
    doc =[0]*tf_data[0].shape[1]
    for match in iterator:
        x=match.span()[0]+1
        y = match.span()[1]-1
        if x==y:
            num = int(f[x])
        else:
            num = int(f[x:y])
        doc[num]=doc[num]+1
    d = tf_data[0].shape[0]
    doc[:] = [math.log(d) if x==0 else math.log(d /x) for x in doc]
    TFIDF = tf_data[0].multiply(doc)
    dump_svmlight_file(TFIDF,tf_data[1], st + "_TFIDF.feat")

# function calls are here

data_train, lines_train ,f_train = get_data("F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\aclImdb\\train\\labeledBow.feat")
data_test, lines_test ,f_test = get_data("F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\aclImdb\\test\\labeledBow.feat")
BBOW(p,data_train,lines_train,"train")
print("BBOW is ready for training data")
BBOW(p,data_test,lines_test,"test")
print("BBOW is ready for testing data")
tf(p,lines_train,data_train,"train")
print("Term frequency is ready for training data")
tf(p,lines_test,data_test,"test")
print("Term frequency is ready for testing data")
tfidf(q,"train",f_train)
print("TFIDF is ready for training data")
tfidf(q,"test",f_test)
print("TFIDF is ready for testing data")

# word2vec is implemented here

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding="utf-8")
    # read all text
    text = file.read()
    file.close()
    return text
# load all docs in a directory
def process_docs(directory):
    lines = ""
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue
        # create the full path of the file to open
        path = directory + '\\' + filename
        # load and clean the doc
        doc = load_doc(path)
        lines = lines +'\n' + doc
        # add to list
    return lines

def train_data(directory):
    #prepare negative reviews
    negative_lines = process_docs(directory + '\\neg')
    # prepare positive reviews
    positive_lines = process_docs(directory + '\\pos')
    return (negative_lines + '\n' + positive_lines)

final_data = train_data('F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\aclImdb\\train') 
sent = sent_tokenize(final_data) 
for idx, s in enumerate(sent):
    sent[idx] = word_tokenize(s)
model = gensim.models.Word2Vec(sent, workers=4,size=300,min_count=1)
model.save("word2vec_vectors")

print("Trained Word vectors are ready for use")


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,[self.labels_list[idx]]) 
    
# load all docs in a directory
def process_docs1(directory):
    data = []
    Labels = []
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue
        # create the full path of the file to open
        path = directory + '\\' + filename
        doc = load_doc(path)
        data.append(doc)
        Labels.append(filename)
    return (data , Labels)
def train_data1(directory):
    #prepare negative reviews
    neg_data , neg_label = process_docs1(directory + '\\neg')
    # prepare positive reviews
    pos_data , pos_label = process_docs1(directory + '\\pos')
    for i in range(0,len(neg_label)):
        pos_data.append(neg_data[i])
        pos_label.append(neg_label[i])
    return (pos_data,pos_label)

final_data,final_label = train_data1('F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\aclImdb\\train') 
it = LabeledLineSentence(final_data, final_label)

model = gensim.models.Doc2Vec(size=300, min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)
model.train(it,total_examples = len(final_data),epochs = 100)
model.save("doc2vec_model")

print("Paragraph vectors ready for use")