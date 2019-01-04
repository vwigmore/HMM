import hidden_markov
import numpy as np
import EasyBid


class ExHMM(hidden_markov.hmm):

    def __init__(self, start_prob=None, trans_prob=None, em_prob=None):
        states = ['a', 'b', 'c', 'd', 'e', 'f']
        observations = EasyBid.bidlabels
        start_prob = np.matrix(np.full((1, len(states)), 1.0/len(states))) if start_prob is None else start_prob
        trans_prob = np.matrix(np.full((len(states), len(states)), 1.0/len(states))) if trans_prob is None else trans_prob
        em_prob = np.matrix(np.full((len(states), len(observations)), 1.0/len(observations))) if em_prob is None else em_prob
        hidden_markov.hmm.__init__(self, states, observations, start_prob, trans_prob, em_prob)

    def train_hmm(self, observation_list, iterations, quantities):
        obs_size = len(observation_list)
        prob = float('inf')
        q = quantities

        # Train the model 'iteration' number of times
        # store em_prob and trans_prob copies since you should use same values for one loop
        for i in range(iterations):

            emProbNew = np.asmatrix(np.zeros((self.em_prob.shape)))
            transProbNew = np.asmatrix(np.zeros((self.trans_prob.shape)))
            startProbNew = np.asmatrix(np.zeros((self.start_prob.shape)))

            for j in range(obs_size):
                # re-assing values based on weight
                emProbNew = emProbNew + q[j] * self._train_emission(observation_list[j])
                transProbNew = transProbNew + q[j] * self._train_transition(observation_list[j])
                startProbNew = startProbNew + q[j] * self._train_start_prob(observation_list[j])

            # Normalizing
            em_norm = emProbNew.sum(axis=1)
            trans_norm = transProbNew.sum(axis=1)
            start_norm = startProbNew.sum(axis=1)

            emProbNew = emProbNew / em_norm
            startProbNew = startProbNew / start_norm
            transProbNew = transProbNew / trans_norm

            self.em_prob, self.trans_prob = emProbNew, transProbNew
            self.start_prob = startProbNew

            if prob - self.log_prob(observation_list, quantities) > 0.0000001:
                prob = self.log_prob(observation_list, quantities)
            else:
                return self.em_prob, self.trans_prob, self.start_prob

        return self.em_prob, self.trans_prob, self.start_prob
