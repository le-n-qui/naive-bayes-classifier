# Qui Le
# 02/05/2018
# Use sys to get command line arguments
import sys
import nltk
import pickle

from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist

# First argument, sys.argv[0], is 
# nbtrain2.py in command line

# Second argument in the command line
corpus_root = sys.argv[1] 
# Third argument in the command line
filename = sys.argv[2]

# This function puts all spams into one list
# and all hams into another
def label_text(label, given_list):
	new_list = []
	for item in given_list:
		if label in item:
			new_list.append(item)
	# Class count
	class_count = len(new_list) 
	# Make a corpus for each class
	class_corpus = PlaintextCorpusReader(corpus_root, new_list)
	# Make a bag of words for each class
	class_bag = class_corpus.words()
	# Return a tuple containing the count 
	# and bag of words for class 
	return (class_count, class_bag)

# This function creates a dictionary 
# for each incidvidual class
def class_dict(key, value):
	# Get the class count
	class_count = value[0]
	# Get the frequecy distribution for each class	
	class_fd = FreqDist(value[1])
	# Make dictionary
	dictionary = { 'label': key, 'count': class_count, 'fd': class_fd }
	return dictionary



# Loading SPAM_training data 
my_corpus = PlaintextCorpusReader(corpus_root, '.*')


# Create a list of plain txt in SPAM_training
samples_list = my_corpus.fileids()

# Create an empty list to save class labels  
label_list = []
for item in samples_list:
	tok_list = item.split('.')
	label_list.append(tok_list[0])

# Remove any repeating label from the list
label_set = set(label_list)

# An empty dictionary whose
# key is label and value is 
# a tuple with two elements 
my_class_dict = {}

# There is a tuple associated 
# with each label (called tag in for loop)
# in label_set
for tag in sorted(label_set):
	my_class_dict[tag] = label_text(tag, samples_list)
		

# an empty list that will be 
# used to save dictionary for
# each class
model = []

# For each key in dictionary my_class_dict
# pass key and value to function
# class_dict to create
# a dictionary for each class
# The returned dictionary will
# be appended to the list called model
for key in my_class_dict.keys():
	model.append(class_dict(key, my_class_dict[key]))

# use pickle.dump to save model
pickle.dump(model, open(filename, 'wb'))

# Below code: test to see the result
storedlist = pickle.load(open(filename, 'rb'))
print(storedlist)