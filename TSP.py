import numpy as np # array operations
import random # stochastic rotue generation
import operator # for fancy sorting stuff
import pandas as pd
import matplotlib.pyplot as plt # yes

class City:
    def __init__(self, n, x, y):
        self.n = int(n)
        self.x = int(x)
        self.y = int(y)

    def distance_from(self, other):
        return np.sqrt(abs(self.x - other.x) ** 2 + abs(self.y - other.y) ** 2)

    def __repr__(self):
        return "City {}: ({}, {})".format(self.n, self.x, self.y)

def inv_l2(route):
    l2 = 0
    for loc_a, loc_b in zip(route[:-1], route[1:]):
        l2 += loc_a.distance_from(loc_b)
    return 1 / l2

class TSPProblem:
    def __init__(self, city_file):
        self.cities = []
        with open(city_file) as f:
            for line in f:
                if line[0] is not '#':
                    city_num, city_x, city_y = line.split()
                    self.cities.append(City(city_num, city_x, city_y))

    def createRoute(self):
        return random.sample(self.cities, len(self.cities))

    def createPopulation(self, pop_size):
        return [self.createRoute() for _ in range(pop_size)]
    
    def rankRoutes(self, population, fitness_func=inv_l2):
        return [fitness_func(population[i]) for i in range(len(population))]

    def resample(self, routes, rankings, sample_size):
        return np.array(routes)[np.random.choice(len(routes), size=sample_size, p=[w / sum(rankings) for w in rankings]), :]

    def breed(self, parent1, parent2):
        loc = random.choice(range(len(parent1))) 
        gene = parent1[loc : random.choice(range(len(parent1)))]
        child = [None] * len(parent1) # sentinel value
        for i in range(len(gene)):
            child[i + loc] = gene[i]
        parent2_iter = 0
        for i in range(len(child)):
            if child[i] is None:
                while True:
                    if parent2[parent2_iter] not in gene:
                        child[i] = parent2[parent2_iter]
                        break
                    parent2_iter += 1       
        return child

    def crossover(self, routes, n_elite):
        next_gen = routes[:n_elite]
        n_children = len(routes) - n_elite
        for _ in range(n_children):
            parent1, parent2 = random.sample(routes, 2)
            next_gen.append(self.breed(parent1, parent2))
        return next_gen

    def mutate(self, routes, p):
        if random.random() < p:
            src, dest = random.sample(range(len(routes)), 2)
            temp = routes[dest]
            routes[dest] = routes[src]
            routes[src] = temp
        return routes

    def iterate(self, curr_gen, n_elite=10, mut_rate=0.001):
        ranked = self.rankRoutes(curr_gen)
        fittest = self.resample(curr_gen, ranked, n_elite)
        crossed = self.crossover(fittest, n_elite)
        mutated = self.mutate(crossed, mut_rate)
        return mutated

    def run(self, n_iters=1000, n_elite=10, mut_rate=0.001, pop_size=100, epsilon=0.0001):
        pop = self.createPopulation(pop_size)
        last_max = None
        for i in range(n_iters):
            if n_iters >= 5000 and i % 100 == 0 and i != 0:
                print("Performing iteration {}/{}".format(i + 1, n_iters))
            pop = self.iterate(pop, n_elite, mut_rate)
            if i > 0:
                if last_max is not None:
                    if epsilon > (max(self.rankRoutes(pop)) - last_max):
                        break # convergence
                last_max = max(self.rankRoutes(pop))
        fitness, best_route = self.report(pop)
        self.plot(fitness, best_route)

    def report(self, population, fitness_func=inv_l2):
        rankings = self.rankRoutes(population)
        best_route = population[rankings.index(max(rankings))]
        print("Length:", 1 / max(rankings))
        print("Route:", str(list(best_route)))
        return 1 / max(rankings), list(best_route)

    def plot(self, fitness, route):
        x = [city.x for city in route]
        y = [city.y for city in route]
        plt.title('Traveling Salesman Solution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.scatter(x, y)
        plt.plot(x, y, '-o')
        plt.text(0, max(y), "Length: "  + str(fitness), horizontalalignment='left', verticalalignment='center')
        plt.show()
            

if __name__ == '__main__':
    tsp = TSPProblem('maps/small_city')
    tsp.run(n_iters=1000, n_elite=100, mut_rate=0.001, pop_size=5000)



