# Qui Le
# 02/18/2018
# Text Classification Bonus

import nltk
import pickle
import sys
from math import log
from nltk.corpus import PlaintextCorpusReader

# Create a function to find the f-score
def fscore(precision, recall):
	score = (2 * precision * recall) / (precision + recall)
	return score

# For multi-class naives bayes classifier
def calculate(document_tokens, my_model, freq_dist, training_corpus_size):
	# Create a variable containing emptry string
	# label_predicted will contain the prediction
	label_predicted = " "
	# Create a variable that will 
	# hold the highest value
	result = 0
	# Create a temporary holder
	temp = 0
	
	# We first find P(c1)P(d|c1) for the first class and assign it to temp
	# Add log of the prior probability before going through each token in document
	temp += log( my_model[0]['count'] / training_corpus_size)
	for token in document_tokens:
		temp += log( (my_model[0]['fd'][token] + 1) / (my_model[0]['fd'].N() + freq_dist.B()) )	
	# Keep temp in result for later comparison
	result = temp
	# Update label_predicted
	label_predicted = my_model[0]['label']
		
	# We now find P(c)P(d|c) for the rest of the classes
	for i in range(1, len(my_model)):
		# Re-initialize temp to zero
		temp = 0
		# Add log of the prior probability
		temp += log( my_model[i]['count'] / training_corpus_size)
		for token in document_tokens:
			temp += log( (my_model[i]['fd'][token] + 1) / (my_model[i]['fd'].N() + freq_dist.B()) )
		# If temp is greater than result
		# result will be updated and 
		# label_predicted will be updated
		# this way, we will find the highest value 
		# for c_pred, the most likely class
		if temp > result:
			result = temp
			label_predicted = my_model[i]['label']

	return label_predicted


# First argument in the command line
# is nbtest2.py (i.e. sys.argv[0])

# Second argument in the command line
# is the directory containing the test
# documents to be classified
corpus_root = sys.argv[1]

# Third argument in the command line
# is the name of the file created by nbtrain2.py
filename = sys.argv[2]

# Create a corpus
my_corpus = PlaintextCorpusReader(corpus_root, '.*')

# restore model from nbtrain2.py
with open(filename, 'rb') as f:
	model = pickle.load(f)

# Get one frequency distribution from multiple classes
fd = model[0]['fd']
for i in range(1, len(model)):
	fd += model[i]['fd'] 

# Get total number of sample documents from model
total_samples = 0
for i in range(0, len(model)):
	total_samples += model[i]['count']

# Different types of counter 
#correct_prediction = 0
#belongs_in_ham = 0
#belongs_in_spam = 0
#classified_as_ham = 0
#classified_as_spam = 0
#correct_ham = 0
#correct_spam = 0

for fileid in my_corpus.fileids():
	# Comment codes below counts how many hams are in training corpus
	#if fileid.startswith(model[0]['label']):
	#	belongs_in_ham += 1
	# Here: how many spams are in training corpus
	#if fileid.startswith(model[1]['label']):
	#	belongs_in_spam += 1

	# Get all tokens from each document
	tokens = my_corpus.words(fileid)
	# Function calculate() returns the prediction 
	# for the document currently being looked at
	prediction = calculate(tokens, model, fd, total_samples)
	# Print out the name of the document and the predicted label
	print(fileid, prediction)

	# Here: More on keeping the count for
	# how many is classified as ham
	#if prediction == model[0]['label']:
	#	classified_as_ham += 1
	# below: how many is classified as spam
	#if prediction == model[1]['label']:
	#	classified_as_spam += 1
	#if fileid.startswith(prediction):
		# how many correct predictions are made
	#	correct_prediction += 1
	#	if prediction == model[0]['label']:
			# how many documents are correctly labeled ham
	#		correct_ham += 1
	#	else: 
			# how many documents are correctly labeled spam
	#		correct_spam += 1

# run nbtrain2.py on SPAM_train folder
# Work on classifier with SPAM_dev folder
# Test classifier with SPAM_test folder 
# Commented out below is code that print out statistics for SPAM_dev
#accuracy = (correct_prediction / len(my_corpus.fileids())) * 100
#print("\nAccuracy rate: ", round(accuracy, 3))
#ham_precision = (correct_ham / classified_as_ham) * 100
#spam_precision = (correct_spam / classified_as_spam) * 100
#ham_recall = (correct_ham / belongs_in_ham) * 100
#spam_recall = (correct_spam / belongs_in_spam) * 100
#print("\nHam precision rate: ", round(ham_precision, 3))
#print("Spam precision rate: ", round(spam_precision, 3))
#print("\nHam recall rate: ", round(ham_recall, 3))
#print("Spam recall rate: ", round(spam_recall, 3))
#print("\nF-score for ham: ", round(fscore(ham_precision, ham_recall), 3))
#print("F-score for spam: ", round(fscore(spam_precision, spam_recall), 3))
