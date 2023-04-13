# deep models for trainable EA
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np


# defines the arguments for training a deep model
class ModelArgs:
    def __init__(self, gene_length, bag_size, noise_size=10, depth=5, batch_size=32, device='cpu'):
        self.gene_length = gene_length
        self.bag_size = bag_size
        # 2 crossover cut positions, 1 mutation rate for each gene
        self.feat_size = bag_size * gene_length * 2 + bag_size
        self.crossover_size = bag_size * gene_length * 2
        self.mutate_size = bag_size
        self.noise_size = noise_size
        self.depth = depth
        self.batch_size = batch_size
        self.device = device


# dataset definition
class BagDataset(Dataset):
    def __init__(self):
        self.crossover_probs = []
        self.mutate_probs = []
        self.fitness_deltas = []
        self.bag_n = 0

    def add_bag(self, gene_bag, fitness_delta):
        cps = []
        mps = []
        for g in gene_bag:
            cps.append(torch.tensor(g.crossover_probs, dtype=torch.float64))
            mps.append(g.mutate_prob)
        cps = torch.stack(cps)
        mps = torch.tensor(mps, dtype=torch.float64)
        self.crossover_probs.append(cps)
        self.mutate_probs.append(mps)
        self.fitness_deltas.append(fitness_delta)
        self.bag_n += 1

    def __len__(self):
        return self.bag_n

    def __getitem__(self, idx):
        cps = self.crossover_probs[idx]
        mps = self.mutate_probs[idx]
        # shuffle bag distributions for data augmentation
        shuffle_idx = np.random.permutation(np.arange(cps.shape[0]))
        cps = cps[shuffle_idx]
        mps = mps[shuffle_idx]
        feats = np.concatenate([cps.reshape(-1), mps])
        fitness_delta = self.fitness_deltas[idx]
        return feats, fitness_delta


# MLP Discriminator
class MLPDis(nn.Module):
    def __init__(self, model_args):
        super(MLPDis, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Linear0',
                               nn.Linear(model_args.feat_size, model_args.feat_size // 4, device=model_args.device))
        self.layers.add_module('LeakyReLU0', nn.LeakyReLU())
        for i in range(model_args.depth - 1):
            self.layers.add_module('Linear' + str(i + 1),
                                   nn.Linear(model_args.feat_size // ((i + 1) * 4),
                                             model_args.feat_size // ((i + 2) * 4),
                                             device=model_args.device))
            self.layers.add_module('LeakyRelu' + str(i + 1), nn.ReLU())
        self.layers.add_module('Linear' + str(model_args.depth + 1),
                               nn.Linear(model_args.feat_size // (model_args.depth * 4), 1, device=model_args.device))
        # normalize for scoring regression
        self.layers.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        return self.layers.forward(x)


# MLP Generator
class MLPGen(nn.Module):
    def __init__(self, model_args):
        super(MLPGen, self).__init__()
        self.bag_size = model_args.bag_size
        self.gene_length = model_args.gene_length

        # crossover layers
        self.crossover_layers = nn.Sequential()
        self.crossover_layers.add_module('Linear0',
                                         nn.Linear(model_args.noise_size, model_args.noise_size * 4,
                                                   device=model_args.device))
        self.crossover_layers.add_module('LeakyReLU0', nn.LeakyReLU())
        for i in range(model_args.depth - 1):
            self.crossover_layers.add_module('Linear' + str(i + 1),
                                             nn.Linear(model_args.noise_size * (i + 1) * 4,
                                                       model_args.noise_size * (i + 2) * 4,
                                                       device=model_args.device))
            self.crossover_layers.add_module('LeakyReLU' + str(i + 1), nn.LeakyReLU())
            # self.crossover_layers.add_module('DropOut' + str(i+1), nn.Dropout(p=.2))
        self.crossover_layers.add_module('Linear' + str(model_args.depth + 1),
                                         nn.Linear(model_args.noise_size * model_args.depth * 4,
                                                   model_args.crossover_size,
                                                   device=model_args.device))
        self.crossover_layers.add_module('ReLU', nn.ReLU())
        # self.crossover_layers.add_module('Sigmoid', nn.Sigmoid())

        # mutation layers
        self.mutation_layers = nn.Sequential()
        self.mutation_layers.add_module('Linear0',
                                        nn.Linear(model_args.noise_size, model_args.noise_size * 2,
                                                  device=model_args.device))
        self.mutation_layers.add_module('LeakyReLU0', nn.LeakyReLU())
        for i in range(model_args.depth - 1):
            self.mutation_layers.add_module('Linear' + str(i + 1),
                                            nn.Linear(model_args.noise_size * (i + 1) * 2,
                                                      model_args.noise_size * (i + 2) * 2,
                                                      device=model_args.device))
            self.mutation_layers.add_module('LeakyReLU' + str(i + 1), nn.LeakyReLU())
            # self.mutation_layers.add_module('DropOut' + str(i + 1), nn.Dropout(p=.2))
        self.mutation_layers.add_module('Linear' + str(model_args.depth + 1),
                                        nn.Linear(model_args.noise_size * model_args.depth * 2,
                                                  model_args.mutate_size,
                                                  device=model_args.device))

    def forward(self, x):
        y1 = self.crossover_layers(x).reshape(x.shape[0], self.bag_size, self.gene_length * 2)
        # normalize for proper probabilities
        y1a = y1[:, :, :self.gene_length] / torch.sum(y1[:, :, :self.gene_length], dim=2).reshape(
            x.shape[0], self.bag_size, 1)
        y1b = y1[:, :, self.gene_length:] / torch.sum(y1[:, :, self.gene_length:], dim=2).reshape(
            x.shape[0], self.bag_size, 1)
        y3 = torch.cat([y1a, y1b], dim=2)
        y2 = torch.sigmoid(self.mutation_layers(x))

        return torch.cat([y3.reshape(x.shape[0], -1), y2], dim=1) * 1e-3  # add eps for nonzero probs
