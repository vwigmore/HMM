import json
import numpy as np
from UtilityFunction import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from ExtHMM import *
import os

# Hardcoded set of possible strategy labels
labels = ["conceder", "random", "hardheaded", "tft"]
hmm = ExtHMM()


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
 
# features X: own utility, opponent utility, chosen discrete values 
# y labels categorically encoded
def loadTrainingData(file_name):
	(agent1_label, agent2_label) = parseAgentLabels(file_name)
	
	# Read the JSON file
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
		
		
		# One hot encode the categorical bids 
		X_bids = np.append(agent1_strat_bids, agent2_strat_bids, axis=0)
		one_hot_enc = OneHotEncoder(categories=issues)
		X_bids = one_hot_enc.fit_transform(X_bids).toarray()
		
		# Append the feature matrices
		X = np.append(agent1_strat_utilities, agent2_strat_utilities, axis=0)
		X = np.append(X, X_bids, axis=1)
		
		# Label encode the ground truth labels
		y = np.append(agent1_y, agent2_y, axis=0)
		label_enc = LabelEncoder()
		label_enc.classes_ = labels
		y = label_enc.fit_transform(y)
		
		return (X, y)
		
# receives issues dictionary from JSON and sets it up for Scikit-learn one hot encoder
def setupIssues(issues):
	categories = []
	for key in sorted(issues.keys()):
		categories.append(issues[key])
	return categories

def main():
	path = "train/"
	files = os.listdir(path)
	for file in files:
		print(path + file)
		(X, y) = loadTrainingData(path + file)
		# Use one hot encoder
		print(X)
		print(y)

		# Normalize

		# Input into HMM
		hmm.fit(X, y, len(y))


if __name__ == "__main__":
    main()