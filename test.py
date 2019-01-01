import json
from main import *
import sys


# return x: feature matrix
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
            # skip final round
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

        return X


def test(file_name):
    X = loadTestData(file_name)
    print(X)

    hmm.predict(X)
    print(hmm.predict(X))


if __name__ == "__main__":
    test(sys.argv[1])