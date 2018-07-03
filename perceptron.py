from random import seed
from random import randrange
import csv
import numpy as np

# Load csv and convert strings into numerical values
from numpy import recfromcsv
from numpy import genfromtxt
my_data = recfromcsv('sonar.csv', delimiter=',')
train = []
for row in my_data:
    train.append(list(row))
for row in train:
    if row[-1] == 'R':
        row[-1] = 0
    elif row[-1] == 'M':
        row[-1] = 1

# Predict the output value given a set of weights
def predict(row,weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation = activation + weights[i+1]*row[i] # this is the weight vector dotted with the row of attributes
    return 1.0 if activation >=0.0 else 0.0 # returns a binary classification

# Estimate starting weights using stochastic grad descent
def train_weights(train,l_rate,n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row,weights)
            error = row[-1] -prediction
            sum_error = sum_error + error**2
            weights[0] = weights[0] + l_rate*error
            for i in range(len(row)-1):
                weights[i+1] = weights[i+1]+l_rate*error*row[i] 
    return weights

# Perceptron
def perceptron(train, test, l_rate, n_epoch):
    predictions = list()
    weights = train_weights(train,l_rate,n_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

n_folds = 3
l_rate = 0.1
n_epoch = 100
scores = evaluate_algorithm(train, perceptron, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
