import numpy as np
from enum import Enum
from itertools import combinations
from random import shuffle, random, randint, seed
from src.ea.gene import Gene
from src.problems.tsp import TSP


class OptimizerMode(Enum):
    VANILLA = 0
    MLP = 1


class Optimizer:
    def __init__(self, problem, mode=OptimizerMode.VANILLA, population=1000,
                 candidate_buffer=2, max_mutate_rate=0.01, stop_delta=1e-6,
                 verbose=False, report_generation=10):
        self.problem = problem
        self.mode = mode
        self.population = population
        self.candidate_buffer = candidate_buffer
        self.max_mutate_rate = max_mutate_rate
        self.stop_delta = stop_delta
        self.verbose = verbose
        self.genes = []
        # meta data for reporting
        self.loss = 1e6
        self.best_candidate = None
        self.generation = 0
        self.report_generation = report_generation

    def fit(self):
        delta = self.stop_delta + 1e6

        # initialisation of gene pool
        for i in range(self.population):
            self.genes.append(Gene(self.problem.gene_length, self.problem.gene_alphabet_length,
                                   self.problem.gene_alphabet_repeat))
        print('Initialized EA: {} population, {} mode.\n'.format(self.population, self.mode.name))

        # main iteration loop
        prev_loss = 0
        print('Evolving')
        while delta > self.stop_delta:
            solution = self.solve()
            loss = self.problem.get_loss(solution)
            delta = abs(prev_loss - loss)
            prev_loss = loss
            if self.generation % self.report_generation == 0 and self.verbose:
                self.report()
            if not self.verbose:
                print('.', end='')
        print('\nConvergence reached:\n')
        self.report()

    def solve(self):  # get solution for one generation
        # select best candidates
        ls = []
        for gene in self.genes:
            gene.fitness = -self.problem.get_loss(self.problem.gene_to_solution(gene))
            ls.append(gene.fitness)
        top_k_idx = np.argsort(ls)[-self.candidate_buffer:]
        best_candidates = [self.genes[i] for i in top_k_idx]

        # populate next generation with crossover and mutation
        self.crossover(best_candidates)
        self.mutate(best_candidates)

        # update meta data
        self.generation += 1
        self.best_candidate = best_candidates[-1]
        self.loss = -self.best_candidate.fitness

        # return solution
        return self.problem.gene_to_solution(self.best_candidate)

    def crossover(self, best_candidates):
        sn = self.candidate_buffer  # spawn number
        combs = list(combinations(np.arange(len(best_candidates)), 2))
        # keep best parent candidates in next generation
        for i in range(self.candidate_buffer):
            self.genes[i] = best_candidates[i]
        # crossover
        while sn < self.population:
            if sn % len(combs) == 0:
                shuffle(combs)
            g1, g2 = combs[sn % len(combs)]
            g1 = best_candidates[g1].sequence
            g2 = best_candidates[g2].sequence
            offspring = None

            # vanilla EA uses a random crossover operator
            if self.mode == OptimizerMode.VANILLA:
                if self.problem.gene_alphabet_repeat:
                    g = np.array(g1)
                    g2_idx = np.random.permutation(np.arange(len(g2)))[:len(g2) // 2 + 1]
                    g[g2_idx] = np.array(g2)[g2_idx]
                    offspring = g
                else:  # ordered crossover
                    g = []
                    ga = int(random() * len(g1))
                    gb = int(random() * len(g1))
                    for i in range(min(ga, gb), max(ga, gb)):
                        g.append(g1[i])
                    g_rest = [n for n in g2 if n not in g]
                    g = g + g_rest
                    offspring = g

            # MLP EA uses a trainable crossover operator based on MLP
            elif self.mode == OptimizerMode.MLP:
                pass

            self.genes[sn] = Gene(offspring)
            sn += 1

    def mutate(self, best_candidates):
        sn = self.candidate_buffer  # spawn number
        # keep best parent candidates in next generation
        for i in range(self.candidate_buffer):
            self.genes[i] = best_candidates[i]
        # mutate the rest
        while sn < self.population:
            g = self.genes[sn]
            # vanilla EA uses a random mutation operator
            if self.mode == OptimizerMode.VANILLA:
                if self.problem.gene_alphabet_repeat:
                    for j in range(len(g)):
                        if random() <= self.max_mutate_rate * random():
                            g.sequence[j] = randint(0, self.problem.gene_alphabet_length - 1)
                else:  # swap mutation
                    for j in range(len(g.sequence)):
                        if random() <= self.max_mutate_rate * random():
                            swap_id = j
                            while swap_id == j:
                                swap_id = randint(0, len(g.sequence) - 1)
                            n = g.sequence[swap_id]
                            g.sequence[swap_id] = g.sequence[j]
                            g.sequence[j] = n

            # MLP EA uses a trainable mutation operator based on MLP
            elif self.mode == OptimizerMode.MLP:
                pass

            self.genes[sn] = g
            sn += 1

    def report(self):
        print('Current Generation: {}\n'.format(self.generation))
        print('Best Candidate:\n{}\n'.format(self.best_candidate))
        print('Loss: {}\n'.format(self.loss))


# test entry point
if __name__ == '__main__':
    # seeding for reproduction
    # seed(5260)
    # np.random.seed(5260)
    # test for vanilla solver
    tsp_p = TSP()
    optimizer = Optimizer(tsp_p, verbose=True)
    optimizer.fit()
