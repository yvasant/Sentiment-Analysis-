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

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,[self.labels_list[idx]])

def load_doc(filename):
	# open the file as read only
    file = open(filename, 'r', encoding="utf-8")
    # read all text
    text = file.read()
    file.close()
    return text
# load all docs in a directory
def process_docs(directory):
    data = []
    Labels = []
	# walk through all files in the folder
    for filename in listdir(directory):
		# skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue
		# create the full path of the file to open
        path = directory + '\\' + filename
		# load and clean the doc
        doc = load_doc(path)
        data.append(doc)
        Labels.append(filename)
		# add to list
    return (data , Labels)
def train_data(directory):
	#prepare negative reviews
    neg_data , neg_label = process_docs(directory + '\\neg')
	# prepare positive reviews
    pos_data , pos_label = process_docs(directory + '\\pos')
    for i in range(0,len(neg_label)):
        pos_data.append(neg_data[i])
        pos_label.append(pos_label[i])
    return (pos_data,pos_label)

final_data,final_label = train_data('F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\aclImdb\\train') 
it = LabeledLineSentence(final_data, final_label)

model = gensim.models.Doc2Vec(size=300, min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)
model.train(it,total_examples = len(final_data),epochs = 100)
model.save("doc2vec_model")