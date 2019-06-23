import numpy as np # array operations
import random # stochastic route generation
import argparse 
import sys
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
        loc1 = random.choice(range(len(parent1))) 
        loc2 = random.choice(range(len(parent1)))
        gene = parent1[loc1 : loc2]
        child = [None] * len(parent1) # sentinel value
        for i in range(loc1, loc2, 1):
            child[i] = gene[i]
        parent2_iter = -1
        for i in range(len(child)):
            if child[i] is None:
                while True:
                    parent2_iter += 1 
                    if parent2[parent2_iter] not in gene:
                        child[i] = parent2[parent2_iter]
                        break           
            else:
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

    def run(self, n_iters=1000, n_elite=10, mut_rate=0.001, pop_size=100, epsilon=0.0001, report_every=100):
        pop = self.createPopulation(pop_size)
        last_max = None 
        for i in range(n_iters):
            pop = self.iterate(pop, n_elite, mut_rate)
            if i % report_every == 0:
                print("=====Result after", i+1, "iteration(s)=====")
                self.report(pop)
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
    psr = argparse.ArgumentParser()
    psr.add_argument('--file', type=str, default='maps/small_city', help='file with cities and coordinates')
    psr.add_argument('--iterations', type=int, default=1000, help='number of iterations before termination (if no convergence)')
    psr.add_argument('--num-elite', type=int, default=10, help='number of "elite" genes that are selected for reproduction')
    psr.add_argument('--generation', type=int, default=100, help='generation size')
    psr.add_argument('--mutation-rate', type=float, default=0.001, help='mutation rate; probability of a swap mutation in a single generation')
    psr.add_argument('--epsilon', type=float, default=0.0001, help='epsilon; convergence parameter')
    psr.add_argument('--report', type=int, default=100, help='reporting frequency')
    args = psr.parse_args()
    if type(args.iterations) is not int:
        print("ERROR: Number of iterations must be an integer.")
        sys.exit(0)
    if type(args.num_elite) is not int:
        print("ERROR: Number of elite genes must be an integer.")
        sys.exit(0)
    if type(args.generation) is not int:
        print("ERROR: Generation size must be an integer.")
        sys.exit(0)
    if args.generation < args.num_elite:
        print("ERROR: Generation size must be larger than the number of elite genes retained.")
        sys.exit(0)
    if type(args.report) is not int:
        print("ERROR: Reporting frequency must be an integer.")
        sys.exit(0)
    if args.mutation_rate < 0 or args.mutation_rate > 1:
        print("ERROR: Mutation rate must be a probability between 0 and 1, inclusive")
        sys.exit(0)
    if args.epsilon < 0:
        print("ERROR: Epsilon must be positive")
        sys.exit(0)
    tsp = TSPProblem(args.file)
    tsp.run(n_iters=args.iterations, n_elite=args.num_elite, mut_rate=args.mutation_rate, pop_size=args.generation, epsilon=args.epsilon, report_every=args.report)



