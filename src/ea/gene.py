import numpy as np


class Gene:
    def __init__(self, *args):
        if len(args) == 1:  # copy the sequence over
            self.sequence = np.array(args[0])
        else:
            self.spawn(args[0], args[1], args[2])
        self.fitness = None

    def spawn(self, gene_length, alphabet_length, alphabet_repeat=True):
        if alphabet_repeat:
            self.sequence = np.random.randint(alphabet_length, size=gene_length)
        else:
            self.sequence = np.random.permutation(np.arange(alphabet_length))

    def __repr__(self):
        return "Sequence: {}\nFitness(Loss): {:.4f}".format(self.sequence, self.fitness)
