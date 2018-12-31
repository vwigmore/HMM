import seqlearn.hmm
import numpy as np

from sklearn.externals import six
from seqlearn._utils import atleast2d_or_csr, safe_sparse_dot, validate_lengths


class ExtHMM(seqlearn.hmm.MultinomialHMM):

    def __init__(self):
        super(ExtHMM, self).__init__(alpha=0.01)

    def predict(self, X, lengths=None):
        X = atleast2d_or_csr(X)
        scores = safe_sparse_dot(X, self.coef_.T)
        if hasattr(self, "coef_trans_"):
            n_classes = len(self.classes_)
            coef_t = self.coef_trans_.T.reshape(-1, self.coef_trans_.shape[-1])
            trans_scores = safe_sparse_dot(X, coef_t.T)
            trans_scores = trans_scores.reshape(-1, n_classes, n_classes)
        else:
            trans_scores = None

        if lengths is None:
            y = viterbi(scores, trans_scores, self.intercept_trans_,
                       self.intercept_init_[0], self.intercept_final_[0])

        else:
            start, end = validate_lengths(X.shape[0], lengths)
            y = [viterbi(scores[start[i]:end[i]], trans_scores,
                        self.intercept_trans_, self.intercept_init_[0],
                        self.intercept_final_[0])
                 for i in six.moves.xrange(len(lengths))]
            y = np.hstack(y)

        return self.classes_[y]


def viterbi(score, trans_score, b_trans, init, final):
    """First-order Viterbi algorithm.
    Parameters
    ----------
    score : array, shape = (n_samples, n_states)
        Scores per sample/class combination; in a linear model, X * w.T.
        May be overwritten.
    trans_score : array, shape = (n_samples, n_states, n_states), optional
        Scores per sample/transition combination.
    trans : array, shape = (n_states, n_states)
        Transition weights.
    init : array, shape = (n_states,)
    final : array, shape = (n_states,)
        Initial and final state weights.
    References
    ----------
    L. R. Rabiner (1989). A tutorial on hidden Markov models and selected
    applications in speech recognition. Proc. IEEE 77(2):257-286.
    """
    n_samples, n_states = score.shape[0], score.shape[1]
    backp = np.empty((n_samples, n_states), dtype=np.intp)

    for j in range(n_states):
        score[0][j] += init[j]

    # Forward recursion. score is reused as the DP table.
    for i in range(1, n_samples):
        for k in range(n_states):
            maxind = 0
            maxval = float("-inf")
            for j in range(n_states):
                candidate = score[i - 1][j] + b_trans[j][k] + score[i][k]
                if trans_score is not None:
                    candidate += trans_score[i][j][k]
                if candidate > maxval:
                    maxind = j
                    maxval = candidate

            score[i][k] = maxval
            backp[i][k] = maxind

    for j in range(n_states):
        score[n_samples - 1][j] += final[j]

    # Path backtracking
    path = np.empty(n_samples, dtype=np.intp)
    path[n_samples - 1] = score[n_samples - 1, :].argmax()

    for i in range(n_samples - 2, -1, -1):
        path[i] = backp[i + 1][path[i + 1]]
    return path