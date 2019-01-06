import json
import sys
import numpy as np
import os
from ExHMM import ExHMM
from UtilityFunction import *
from EasyBid import *
import pickle

# Hardcoded set of possible strategy labels
labels = ["conceder", "random", "hardheaded", "tft"]


# Method to parse the file name to get the agent strategy labels
def parseAgentLabels(file_name):
	agents = file_name.split('_')
	agent1_label = "None"
	agent2_label = "None"
	for label in labels:
		if label in agents[0]:
			agent1_label = label
		if label in agents[1]:
			agent2_label = label
	return (agent1_label, agent2_label)


def loadData(dir_name, remove_file=None):
	files = []
	for f in os.listdir(dir_name):
		files.append(dir_name + "/" + f)
	if remove_file:
		files.remove(remove_file)
	X0 = []
	X1 = []
	X2 = []
	X3 = []

	for f in files:
		(agent1_label, agent2_label) = parseAgentLabels(f)
		# Read the JSON file
		with open(f, "r") as file:
			data = json.load(file)
			num_bids = len(data['bids'])

			# Normalizing utility function from JSON
			utility1 = data['Utility1']
			utility2 = data['Utility2']

			# Set up data structures to store the features in
			# Last bid is the accept or end of negotiation
			agent1_strat_utilities = np.empty((num_bids - 1, 2))
			agent2_strat_utilities = np.empty((num_bids - 1, 2))

			for index, bid in enumerate(data['bids']):
				if index >= num_bids - 1:
					break
				agent1_bid = bid['agent1'].split(',')
				agent1_strat_utilities[index, 0] = computeUtility(agent1_bid, utility1)
				agent1_strat_utilities[index, 1] = computeUtility(agent1_bid, utility2)

				agent2_bid = bid['agent2'].split(',')
				agent2_strat_utilities[index, 0] = computeUtility(agent2_bid, utility2)
				agent2_strat_utilities[index, 1] = computeUtility(agent2_bid, utility1)

			if agent1_label == labels[0]:
				X0.extend([simplifyBids(agent1_strat_utilities)])
			elif agent1_label == labels[1]:
				X1.extend([simplifyBids(agent1_strat_utilities)])
			elif agent1_label == labels[2]:
				X2.extend([simplifyBids(agent1_strat_utilities)])
			else:
				X3.extend([simplifyBids(agent1_strat_utilities)])

			if agent2_label == labels[0]:
				X0.extend([simplifyBids(agent1_strat_utilities)])
			elif agent2_label == labels[1]:
				X1.extend([simplifyBids(agent1_strat_utilities)])
			elif agent2_label == labels[2]:
				X2.extend([simplifyBids(agent1_strat_utilities)])
			else:
				X3.extend([simplifyBids(agent1_strat_utilities)])

	return X0, X1, X2, X3


def testData(file_name):
	with open(file_name, "r") as file:
		data = json.load(file)
		num_bids = len(data['bids'])
		num_issues = len(data['issues'])
		issues = setupIssues(data['issues'])

		# Normalizing utility function from JSON
		utility1 = data['Utility1']
		utility2 = data['Utility2']

		# Set up data structures to store the features in
		# Last bid is the accept or end of negotiation
		agent1_strat_utilities = np.empty((num_bids - 1, 2))
		agent2_strat_utilities = np.empty((num_bids - 1, 2))

		for index, bid in enumerate(data['bids']):
			if index >= num_bids - 1:
				break
			agent1_bid = bid['agent1'].split(',')
			agent1_strat_utilities[index, 0] = computeUtility(agent1_bid, utility1)
			agent1_strat_utilities[index, 1] = computeUtility(agent1_bid, utility2)

			agent2_bid = bid['agent2'].split(',')
			agent2_strat_utilities[index, 0] = computeUtility(agent2_bid, utility2)
			agent2_strat_utilities[index, 1] = computeUtility(agent2_bid, utility1)

		A1 = simplifyBids(agent1_strat_utilities)
		A2 = simplifyBids(agent2_strat_utilities)
		return A1, A2

		
# receives issues dictionary from JSON and sets it up for Scikit-learn one hot encoder
def firstColumn(array):
	newArray = []
	for i in range(0, len(array)):
		newArray.append(array[i][0])
	newArray = np.reshape(newArray, (-1, 1))
	return newArray


def setupIssues(issues):
	categories = []
	for key in sorted(issues.keys()):
		categories.append(issues[key])
	return categories

def setupLabels(labels):
	categories = []
	for cat in sorted(labels):
		categories.append(cat)
	return [categories]

def main():

	if sys.argv[1] == "train":

		train_directory = sys.argv[2]
		X0, X1, X2, X3 = loadData(train_directory)

		hmm0 = ExHMM()
		hmm0.train_hmm(X0, 100, np.full((len(X0)), 1))
		with open('intermediate/hmm0.pickle', 'wb') as file:
			pickle.dump(hmm0, file)

		hmm1 = ExHMM()
		hmm1.train_hmm(X1, 100, np.full((len(X1)), 1))
		with open('intermediate/hmm1.pickle', 'wb') as file:
			pickle.dump(hmm1, file)

		hmm2 = ExHMM()
		hmm2.train_hmm(X2, 100, np.full((len(X2)), 1))
		with open('intermediate/hmm2.pickle', 'wb') as file:
			pickle.dump(hmm2, file)

		hmm3 = ExHMM()
		hmm3.train_hmm(X3, 100, np.full((len(X3)), 1))
		with open('intermediate/hmm3.pickle', 'wb') as file:
			pickle.dump(hmm3, file)

		print("TRAINING DONE")
	elif sys.argv[1] == "test":

		with open('intermediate/hmm0.pickle', 'rb') as file:
			hmm0test = pickle.load(file)
		with open('intermediate/hmm1.pickle', 'rb') as file:
			hmm1test = pickle.load(file)
		with open('intermediate/hmm2.pickle', 'rb') as file:
			hmm2test = pickle.load(file)
		with open('intermediate/hmm3.pickle', 'rb') as file:
			hmm3test = pickle.load(file)

		test_directory = sys.argv[2]
		test_file = sys.argv[3]
		A, B = testData(test_directory + "/" + test_file)

		# Input into HMM
		catA0 = hmm0test.forward_algo(A)
		catB0 = hmm0test.forward_algo(B)

		catA1 = hmm1test.forward_algo(A)
		catB1 = hmm1test.forward_algo(B)

		catA2 = hmm2test.forward_algo(A)
		catB2 = hmm2test.forward_algo(B)

		catA3 = hmm3test.forward_algo(A)
		catB3 = hmm3test.forward_algo(B)

		Asum = catA0 + catA1 + catA2 + catA3
		Bsum = catB0 + catB1 + catB2 + catB3

		print()
		print("A is conceder", catA0 / Asum)
		print("A is random", catA1 / Asum)
		print("A is hardheaded", catA2 / Asum)
		print("A is tft", catA3 / Asum)
		print("B is conceder", catB0 / Bsum)
		print("B is random", catB1 / Bsum)
		print("B is hardheaded", catB2 / Bsum)
		print("B is tft", catB3 / Bsum)
		print()


	########################## FOR TESTING #############################
	########## Comment above if-else block in main method when testing ##########
	# # k-fold cross validation
	# files = []
	# for f in os.listdir("./train"):
	# 	files.append("./train/" + f)
	# for f in files:
    #
	# 	X0, X1, X2, X3 = loadData("./train", f)
	# 	A, B = testData(f)
	# 	# Use one hot encoder
	# 	# print(X)
	# 	# print(y)
	# 	# print(length)
	# 	# Normalize
    #
	# 	# Input into HMM
	# 	hmm0 = ExHMM()
	# 	e0, t0, s0 = hmm0.train_hmm(X0, 1000, np.full((len(X0)), 1))
	# 	hmm0test = ExHMM(s0, t0, e0)
	# 	catA0 = hmm0test.forward_algo(A)
	# 	catB0 = hmm0test.forward_algo(B)
    #
	# 	hmm1 = ExHMM()
	# 	e1, t1, s1 = hmm1.train_hmm(X1, 1000, np.full((len(X1)), 1))
	# 	hmm1test = ExHMM(s1, t1, e1)
	# 	catA1 = hmm1test.forward_algo(A)
	# 	catB1 = hmm1test.forward_algo(B)
    #
	# 	hmm2 = ExHMM()
	# 	e2, t2, s2 = hmm2.train_hmm(X2, 1000, np.full((len(X2)), 1))
	# 	hmm2test = ExHMM(s2, t2, e2)
	# 	catA2 = hmm2test.forward_algo(A)
	# 	catB2 = hmm2test.forward_algo(B)
    #
	# 	hmm3 = ExHMM()
	# 	e3, t3, s3 = hmm3.train_hmm(X3, 1000, np.full((len(X3)), 1))
	# 	hmm3test = ExHMM(s3, t3, e3)
	# 	catA3 = hmm3test.forward_algo(A)
	# 	catB3 = hmm3test.forward_algo(B)
    #
	# 	Asum = catA0 + catA1 + catA2 + catA3
	# 	Bsum = catB0 + catB1 + catB2 + catB3
    #
	# 	print f
	# 	print "A is conceder", catA0/Asum
	# 	print "A is random", catA1/Asum
	# 	print "A is hardheaded", catA2/Asum
	# 	print "A is tft", catA3/Asum
	# 	print "B is conceder", catB0 / Bsum
	# 	print "B is random", catB1 / Bsum
	# 	print "B is hardheaded", catB2 / Bsum
	# 	print "B is tft", catB3 / Bsum
	# 	print
	########################## FOR TESTING #############################


if __name__ == "__main__":
	main()
