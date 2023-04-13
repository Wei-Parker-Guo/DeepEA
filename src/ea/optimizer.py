from datetime import datetime
import json
import numpy as np
from enum import Enum
from pathlib import Path
from itertools import combinations
from random import shuffle, random, randint, seed
from src.ea.gene import Gene
from src.problems.tsp import TSP
from src.ea.models import ModelArgs, MLPDis, MLPGen, BagDataset
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader


class OptimizerMode(Enum):
    VANILLA = 0
    MLP = 1


class Optimizer:
    def __init__(self, problem, mode=OptimizerMode.VANILLA, population=10000,
                 bags_n=100, noise_size=10, depth=3, batch_size=32, device='cpu',  # DL parameters
                 real_alpha=.7, train_iter=3, converge_epochs=3, converge_delta=1e-5,  # DL parameters
                 candidate_buffer=50, max_mutate_rate=0.01, stop_delta=1e-6, stop_interval=20,
                 verbose=False, report_generation=10):
        self.problem = problem
        self.history = {'generation': [], 'loss': []}
        self.mode = mode
        self.population = population
        self.candidate_buffer = candidate_buffer
        self.max_mutate_rate = max_mutate_rate
        self.stop_delta = stop_delta
        self.stop_interval = stop_interval
        self.verbose = verbose
        self.genes = []
        # initiate components for deep models if mode is deep
        self.bags_n = bags_n
        if mode == OptimizerMode.MLP:
            self.model_args = ModelArgs(gene_length=problem.gene_length, bag_size=population//bags_n,
                                        noise_size=noise_size, depth=depth, batch_size=batch_size, device=device)
            self.mlp_dis = MLPDis(self.model_args)
            self.mlp_gen = MLPGen(self.model_args)
            self.criterion = nn.MSELoss()
            self.mlp_d_optimizer = Adam(self.mlp_dis.parameters())
            self.mlp_g_optimizer = Adam(self.mlp_gen.parameters())
            self.bag_dataset = BagDataset()
            self.real_alpha = real_alpha
            self.train_iter = train_iter
            self.converge_epochs = converge_epochs
            self.converge_delta = converge_delta
            self.gen_feats = None
        # meta data for reporting
        self.loss = 1e6
        self.best_candidate = None
        self.generation = 0
        self.report_generation = report_generation

    def fit(self):
        # initialisation of gene pool
        for i in range(self.population):
            self.genes.append(Gene(self.problem.gene_length, self.problem.gene_alphabet_length,
                                   self.problem.gene_alphabet_repeat))
        print('Initialized EA: {} population, {} elites, {} mode.'.format(
            self.population, self.candidate_buffer, self.mode.name))
        print('Stopping Condition: loss delta less than {} for {} generations.\n'.format(
            self.stop_delta, self.stop_interval))
        print('{}\n'.format(self.problem))

        # main iteration loop
        prev_loss = 0
        print('Evolving')
        # stopping condition: delta is small enough for specified generation interval
        s_interval = 0
        while s_interval < self.stop_interval:
            solution = self.solve()
            loss = self.problem.get_loss(solution)
            delta = abs(prev_loss - loss)
            s_interval = s_interval + 1 if delta < self.stop_delta else 0
            prev_loss = loss
            if self.generation % self.report_generation == 0 and self.verbose:
                self.report()
            if not self.verbose:
                print('.', end='')
            # record history
            self.history['generation'].append(self.generation)
            self.history['loss'].append(self.loss)
        # converged, report results
        timestamp = datetime.now().strftime("[%d-%m-%Y-%H:%M:%S]")
        filename = Path.cwd().joinpath(f'{timestamp}-history.json')
        with open(filename, 'w') as f:
            json.dump(self.history, f)
        print('\nConvergence reached:\n')
        print(f'Results written to {filename}.\n')
        self.report()
        return filename

    def solve(self):  # get solution for one generation
        # select best candidates
        ls = []
        for gene in self.genes:
            gene.fitness = -self.problem.get_loss(self.problem.gene_to_solution(gene))
            ls.append(gene.fitness)
        top_k_idx = np.argsort(ls)[-self.candidate_buffer:]
        best_candidates = [self.genes[i] for i in top_k_idx]

        # train deep models only if needed, ignore initiation round since there is no fitness delta yet
        if self.mode == OptimizerMode.MLP and self.generation > 0 and self.generation % self.train_iter == 0:
            self.mlp_train()
            self.bag_dataset = BagDataset()  # reset dataset for next training

        # generate crossover and mutate probs from deep models if specified
        if self.mode == OptimizerMode.MLP:
            noises = torch.randn(self.bags_n, self.model_args.noise_size)
            self.mlp_gen.eval()
            self.gen_feats = self.mlp_gen(noises.to(self.model_args.device)).cpu().detach().numpy()

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
                self.genes[sn] = Gene(offspring)

            # MLP EA uses a trainable crossover operator based on MLP
            elif self.mode == OptimizerMode.MLP:
                offspring, crossover_probs = self.mlp_crossover(g1, g2, sn)
                self.genes[sn] = Gene(offspring)
                self.genes[sn].crossover_probs = crossover_probs

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
                g = self.mlp_mutate(g, sn)

            self.genes[sn] = g
            sn += 1

    def mlp_train(self):
        # update dataset
        fitness_deltas = []
        for i in range(0, self.population, self.model_args.bag_size):
            gs = self.genes[i:i+self.model_args.bag_size]
            avg_fitness_delta = 0
            for g in gs:
                avg_fitness_delta += self.loss+g.fitness
            avg_fitness_delta /= self.model_args.bag_size
            fitness_deltas.append(avg_fitness_delta)
        # normalise bag scores
        fitness_deltas = np.array(fitness_deltas)
        fitness_deltas = fitness_deltas / fitness_deltas.sum()
        # add to bag
        for i in range(0, self.population, self.model_args.bag_size):
            gs = self.genes[i:i+self.model_args.bag_size]
            self.bag_dataset.add_bag(gs, fitness_deltas[i//self.model_args.bag_size])
        # create loader
        data_loader = DataLoader(self.bag_dataset, batch_size=self.model_args.batch_size, shuffle=True)

        # training
        converge_delta = 1e6
        converge_epochs = 0
        prev_loss = 0
        epoch_n = 1

        if self.verbose:
            print('\nTraining MLP GAN, Generation {}:\n'.format(self.generation))

        # one epoch
        while converge_epochs < self.converge_epochs or converge_delta > self.converge_delta:
            last_loss = 0
            for i, (feats, fitness_deltas) in enumerate(data_loader):
                feats = feats.float()
                fitness_deltas = fitness_deltas.float()

                # train discriminator
                self.mlp_dis.train()
                self.mlp_gen.eval()
                self.mlp_d_optimizer.zero_grad()
                pred_deltas = self.mlp_dis(feats.to(self.model_args.device))
                # fake samples
                noises = torch.randn(self.model_args.batch_size, self.model_args.noise_size)
                gen_deltas = self.mlp_dis(self.mlp_gen(noises.to(self.model_args.device)))
                # loss calculation
                real_loss = self.criterion(pred_deltas.reshape(-1), fitness_deltas.to(self.model_args.device))
                fake_loss = self.criterion(gen_deltas.reshape(-1),
                                           torch.zeros(self.model_args.batch_size).to(self.model_args.device))
                loss = self.real_alpha * real_loss + (1-self.real_alpha) * fake_loss
                last_loss = loss.cpu().item()
                loss.backward()
                self.mlp_d_optimizer.step()

                # train generator
                self.mlp_dis.eval()
                self.mlp_gen.train()
                self.mlp_g_optimizer.zero_grad()
                noises = torch.randn(self.model_args.batch_size, self.model_args.noise_size)
                feats = self.mlp_gen(noises.to(self.model_args.device))
                gen_deltas = self.mlp_dis(feats.to(self.model_args.device))
                target_deltas = torch.ones(gen_deltas.shape[0])
                loss = self.criterion(gen_deltas.reshape(-1), target_deltas.to(self.model_args.device))
                # add a negative log penalty term to punish zero entries in generation
                # loss += -0.2 * torch.log(feats+1e-6).sum()
                loss.backward()
                self.mlp_g_optimizer.step()

            converge_delta = abs(prev_loss-last_loss)
            converge_epochs = converge_epochs+1 if converge_delta <= self.converge_delta else 0
            prev_loss = last_loss

            # reporting
            if self.verbose and epoch_n % 2 == 0:
                print('Epoch {} loss: {}'.format(epoch_n, last_loss))
            elif not self.verbose and epoch_n % 10 == 0:
                print('.', end='')

            epoch_n += 1

        if self.verbose:
            print('Training Complete.\n')

    def mlp_crossover(self, g1, g2, sn):
        # get crossover probs
        bag_n = sn // self.model_args.bag_size
        crossover_probs = self.gen_feats[bag_n][:self.model_args.bag_size * self.model_args.gene_length * 2]
        crossover_probs = crossover_probs.reshape(self.model_args.bag_size, -1)[sn % self.model_args.bag_size]

        # spawn
        offspring = []
        ga_probs = crossover_probs[:self.model_args.gene_length]
        ga_probs = ga_probs / ga_probs.sum()
        gb_probs = crossover_probs[self.model_args.gene_length:]
        gb_probs = gb_probs / gb_probs.sum()
        idx = np.arange(g1.shape[0])
        ga = np.random.choice(g1, p=ga_probs)
        gb = np.random.choice(g1, p=gb_probs)
        for i in range(min(ga, gb), max(ga, gb)):
            offspring.append(g1[i])
        g_rest = [n for n in g2 if n not in offspring]
        offspring = offspring + g_rest

        return offspring, crossover_probs

    def mlp_mutate(self, g, sn):
        # get mutate rate
        bag_n = sn // self.model_args.bag_size
        mutate_probs = self.gen_feats[bag_n][self.model_args.bag_size * self.model_args.gene_length * 2:]
        mutate_rate = mutate_probs[sn % self.model_args.bag_size]

        # spawn
        for i in range(len(g.sequence)):
            if random() <= mutate_rate:
                swap_id = i
                while swap_id == i:
                    swap_id = randint(0, len(g.sequence) - 1)
                n = g.sequence[swap_id]
                g.sequence[swap_id] = g.sequence[i]
                g.sequence[i] = n
        g.mutate_prob = mutate_rate

        return g

    def report(self):
        print('Current Generation: {}\n'.format(self.generation))
        print('Best Candidate:\n{}\n'.format(self.best_candidate))
        print('Loss: {}\n'.format(self.loss))


# test entry point
if __name__ == '__main__':
    # seeding for reproduction
    # seed(5260)
    # np.random.seed(5260)
    # test for MLP solver
    tsp_p = TSP(cities=30)
    optimizer = Optimizer(tsp_p, mode=OptimizerMode.MLP, verbose=True)  # turn on verbose for logging
    optimizer.fit()
