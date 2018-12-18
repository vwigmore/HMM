import seqlearn.hmm

class ExtHMM(seqlearn.hmm.MultinomialHMM):

    def predict(self, X, lengths=None):
        return "test"


if __name__ == '__main__':

    print(ExtHMM().predict(""))
