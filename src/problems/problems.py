import numpy as np


# Abstract Class
class Problem:
    def __init__(self, param_length):
        self.gene_length = param_length
        self.gene_alphabet_length = 42
        self.gene_alphabet_repeat = True

    def get_loss(self, solution):
        return 42

    def gene_to_solution(self, gene):
        return 42
