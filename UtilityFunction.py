
# Normalize the utility function dictionary
def normalizeUtilityFunction(utility_function):
	issue_sum = 0.0
	for k1, issue in utility_function.items():
		issue_sum += issue["weight"]
		sum = 0.0
		for k2, value in issue.items():
			if "weight" not in k2: 
				sum += value
		#Normalize Discrete Values
		for k2, value in issue.items():
			if "weight" not in k2: 
				issue[k2] /= sum
	#Normalize Issue Weights
	for k1, issue in utility_function.items():
		issue["weight"] /= issue_sum
	return utility_function
	
# bid is array of parsed round bid of an agent
# Method computes the utility, assuming the issues are ordered alphabetically
def computeUtility(bid, utility_function):
	utility = 0.0
	for index, key in enumerate(sorted(utility_function.keys())):
		issue = utility_function[key]
		discrete_value = bid[index]
		utility += issue["weight"] * issue[discrete_value]
	return utility
			