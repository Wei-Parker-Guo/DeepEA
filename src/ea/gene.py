import numpy as np
from random import random


class Gene:
    def __init__(self, *args):
        self.fitness = None
        self.crossover_probs = None
        self.mutate_prob = None
        if len(args) == 1:  # copy the sequence over
            self.sequence = np.array(args[0])
        else:
            self.spawn(args[0], args[1], args[2])

    def spawn(self, gene_length, alphabet_length, alphabet_repeat=True):
        if alphabet_repeat:
            self.sequence = np.random.randint(alphabet_length, size=gene_length)
        else:
            self.sequence = np.random.permutation(np.arange(alphabet_length))
        # learnable parameters for DL models
        self.crossover_probs = np.random.rand(gene_length*2)
        self.mutate_prob = random()

    def __repr__(self):
        return "Sequence: {}\nFitness(Loss): {:.4f}\nCrossover Probs: {}\nMutation Rate: {}".format(
            self.sequence, self.fitness, self.crossover_probs, self.mutate_prob)
