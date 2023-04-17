from src.problems.problems import Problem
from random import random
from enum import Enum
import numpy as np


class CityDistrib(Enum):
    UNIFORM = 0
    NORMAL = 1


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


# The Travelling Salesman Problem
class TSP(Problem):
    def __init__(self, cities=50, city_distrib=CityDistrib.NORMAL):
        super().__init__(param_length=cities)
        self.gene_alphabet_length = cities
        self.gene_alphabet_repeat = False
        # initialize cities
        self.city_distrib = city_distrib
        self.cities = []
        for i in range(cities):
            city = None
            if city_distrib == CityDistrib.UNIFORM:
                city = City(random(), random())
            elif city_distrib == CityDistrib.NORMAL:
                x, y = np.random.randn(1)[0], np.random.randn(1)[0]
                city = City(np.clip(x, -3, 3), np.clip(y, -3, 3))
            self.cities.append(city)

    def get_loss(self, solution):
        total_distance = 0
        for i in range(len(solution)-1):
            total_distance += self.cities[solution[i]].distance(self.cities[solution[i+1]])
        # return to start city from end city
        total_distance += self.cities[solution[0]].distance(self.cities[solution[-1]])
        return total_distance

    def gene_to_solution(self, gene):
        return list(gene.sequence)

    def __repr__(self):
        return 'Problem Type: {}\nParameters: {} cities, {} distribution'.format(
            'Travelling Salesman', len(self.cities), self.city_distrib.name)
