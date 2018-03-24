from os import listdir
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
import numpy as np
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
