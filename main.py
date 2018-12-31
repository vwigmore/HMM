import json
import numpy as np
import glob, os
from ExtHMM import ExtHMM
from UtilityFunction import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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

def loadTestData(file_name):

	with open(file_name, "r") as file:
		data = json.load(file)
		num_bids = len(data['bids'])
		num_issues = len(data['issues'])
		issues = setupIssues(data['issues'])

		# Normalizing utility function from JSON
		utility1 = normalizeUtilityFunction(data['Utility1'])
		utility2 = normalizeUtilityFunction(data['Utility2'])

		# Set up data structures to store the features in
		# Last bid is the accept or end of negotiation
		agent1_strat_utilities = np.empty((num_bids - 1, 2))
		agent1_strat_bids = np.empty((num_bids - 1, num_issues), dtype=object)
		agent2_strat_utilities = np.empty((num_bids - 1, 2))
		agent2_strat_bids = np.empty((num_bids - 1, num_issues), dtype=object)

		for index, bid in enumerate(data['bids']):
			if index >= num_bids - 1:
				break
			agent1_bid = bid['agent1'].split(',')
			agent1_strat_utilities[index, 0] = computeUtility(agent1_bid, utility1)
			agent1_strat_utilities[index, 1] = computeUtility(agent1_bid, utility2)
			agent1_strat_bids[index, :] = agent1_bid

			agent2_bid = bid['agent2'].split(',')
			agent2_strat_utilities[index, 0] = computeUtility(agent2_bid, utility2)
			agent2_strat_utilities[index, 1] = computeUtility(agent2_bid, utility1)
			agent2_strat_bids[index, :] = agent2_bid

		# One hot encode the categorical bids
		X_bids = np.append(agent1_strat_bids, agent2_strat_bids, axis=0)
		one_hot_enc = OneHotEncoder(categories=issues)
		X_bids = one_hot_enc.fit_transform(X_bids).toarray()

		# Append the feature matrices
		X = np.append(agent1_strat_utilities, agent2_strat_utilities, axis=0)
		X = np.append(X, X_bids, axis=1)

		return (X, [index, index])


# features X: own utility, opponent utility, chosen discrete values 
# y labels categorically encoded
def loadTrainingData(dir_name, remove_file=None):

	# os.chdir("~./"+dir_name)
	files = []
	for f in glob.glob("*.json"):
		files.append(f)
	if remove_file:
		files.remove(remove_file)

	# initiate arrays
	X_bids = []
	X = []
	y = []
	lengths = []
	for f in files:
		(agent1_label, agent2_label) = parseAgentLabels(f)
		# Read the JSON file
		with open(f, "r") as file:
			data = json.load(file)
			num_bids = len(data['bids'])
			num_issues = len(data['issues'])
			issues = setupIssues(data['issues'])
		
			# Normalizing utility function from JSON
			utility1 = normalizeUtilityFunction(data['Utility1'])
			utility2 = normalizeUtilityFunction(data['Utility2'])
		
			# Set up data structures to store the features in
			# Last bid is the accept or end of negotiation
			agent1_strat_utilities = np.empty((num_bids - 1, 2))
			agent1_strat_bids = np.empty((num_bids - 1, num_issues), dtype=object)
			agent1_y = np.full((num_bids - 1), agent1_label)
			agent2_strat_utilities = np.empty((num_bids - 1, 2))
			agent2_strat_bids = np.empty((num_bids - 1, num_issues), dtype=object)
			agent2_y = np.full((num_bids - 1), agent2_label)
		
			for index, bid in enumerate(data['bids']):
				if index >= num_bids - 1:
					break
				agent1_bid = bid['agent1'].split(',')
				agent1_strat_utilities[index, 0] = computeUtility(agent1_bid, utility1)
				agent1_strat_utilities[index, 1] = computeUtility(agent1_bid, utility2)
				agent1_strat_bids[index, :] = agent1_bid

				agent2_bid = bid['agent2'].split(',')
				agent2_strat_utilities[index, 0] = computeUtility(agent2_bid, utility2)
				agent2_strat_utilities[index, 1] = computeUtility(agent2_bid, utility1)
				agent2_strat_bids[index, :] = agent2_bid

			X_bids = np.append(X_bids, agent1_strat_bids, axis=0) if len(X_bids) else agent1_strat_bids
			X_bids = np.append(X_bids, agent2_strat_bids, axis=0)

			X = np.append(X, agent1_strat_utilities, axis=0) if len(X) else agent1_strat_utilities
			X = np.append(X, agent2_strat_utilities, axis=0)

			y = np.append(y, agent1_y, axis=0) if len(y) else agent1_y
			y = np.append(y, agent2_y, axis=0)

			lengths.append(index)
			lengths.append(index)

	# One hot encode the categorical bids
	one_hot_enc = OneHotEncoder(categories=issues)
	X_bids = one_hot_enc.fit_transform(X_bids).toarray()
		
	# Append the feature matrices
	X = np.append(X, X_bids, axis=1)
		
	# Label encode the ground truth labels
	label_enc = LabelEncoder()
	label_enc.classes_ = labels
	y = label_enc.fit_transform(y)
	return (X, y, lengths, label_enc)
		
# receives issues dictionary from JSON and sets it up for Scikit-learn one hot encoder
def setupIssues(issues):
	categories = []
	for key in sorted(issues.keys()):
		categories.append(issues[key])
	return categories

def main():

	# k-fold cross validation
	os.chdir("./train")
	files = []
	for f in glob.glob("*.json"):
		files.append(f)
	for f in files:

		(X, y, length, label_enc) = loadTrainingData("train", f)
		# Use one hot encoder
		# print(X)
		# print(y)
		# print(length)
		# Normalize

		# Input into HMM
		hmm = ExtHMM()
		hmm.fit(X, y, [length])

		# without directory because of changed dir in
		(T, Tlength) = loadTestData(f)
		classes = hmm.predict(T, Tlength)


		print "file_name:", f
		print "agent1:"
		count1 = [0.0, 0.0, 0.0, 0.0]
		for i in range(0, len(classes)/2):
			count1[classes[i]] += 1
		for i in range(0, len(count1)):
			count1[i] = round(count1[i] * 200.0 / len(classes), 3)
			print(label_enc.classes_[i], count1[i])

		# print("CLASSES", classes)
		# print(len(classes))
		print "agent2:"
		count2 = [0.0, 0.0, 0.0, 0.0]
		for i in range(len(classes)/2, len(classes)):
			count2[classes[i]] += 1
		print(count2)
		for i in range(0, len(count2)):
			count2[i] = round(count2[i] * 200.0 / len(classes), 3)
			print(label_enc.classes_[i], count2[i])
		print
   

if __name__ == "__main__":
    main()