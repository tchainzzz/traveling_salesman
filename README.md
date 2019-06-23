# traveling_salesman

This is an implementation of a Traveling Salesman Problem solver via a genetic algorithm. 

# Problem
Given a vector of n cities, what is the shortest path that connects all cities? This problem is known to be NP-hard; in fact, a brute-force 
naive solution takes O(n!) time, because we need to try every possible ordering of the cities. So is there a faster way?

[Wikipedia gives an awesome overview of this problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem), and also provides some
basic algorithms that are used to solve this. 

# Motivating factors
This problem is often used to benchmark optimization methods. In pure form, this also has implications for route navigation, microchip design,
logistics planning, and other problems that take on the same general form of minimizing the cost of some sequence of actions. 

# Algorithms
A dynamic programming solution [was discovered](https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm)
by Bellman, Held, and Karp, which has complexity O(n²2ⁿ). The recurrence is motivated by the fact that subpaths of shortest paths are also shortest 
paths.

Heuristics like nearest-neighbor, ant colony optimization, and deep graph theory magic that I don't yet understand have also been used.

# Our solution
A genetic algorithm is a randomized algorithm that uses an evolutionary heuristic to traverse the search space. In the context of this problem, 
multiple routes are stochastically generated within the search space. Our implementation generally follows this structure:

Initilaization: Stochastically generate *n* random routes, which are vectors of cities.

Each iteration of this algorithm follows this general template:
1. Given current population (a "generation"), calculate fitness
2. Select fittest members of population
3. Breed via ordered crossover
4. Mutate to produce next generation

Termination: number of iterations has been reached, or the max-fitness (minimum distance in this context) achieved at a particular step 
converges. Convergence occurs when the difference in max-fitness between sequential iterations is less than a specified hyperparameter
epsilon (default 1e-4).

## Fitness function 
The fitness function, by default the reciprocal of the L2 norm 
(Euclidean distance travelled), is used to sample the "fittest" routes with weight proportional to their fitness. 

## Breeding
Within the subset of 
fittest routes, some directly persist into the next generation. Others are replaced with children "breeded" via ordered crossover 
([more technical explanation here](http://www.dmi.unict.it/mpavone/nc-cs/materiale/moscato89.pdf)), in which a subsequence of a parent route is 
randomly selected and "grafted" onto another parent route. More specifically, a random subsequence S of parent 1 is slotted verbatim into an empty
array, and the rest of the slots are filled in-order by elements of parent 2 that do not appear in S.

## Mutation
Lastly, the mutation step occurs. We use swap mutation since this problem has the unique property that each city must appear once in a
route. This means that with some probability p (default 1e-03), two elements in the population are swapped. 

Note that GAs are **not** guaranteed to converge to a global optimum, but in practice, with enough iterations and proper hyperparameter tuning,
it is likely unless the pseudorandom number generator is feeling particularly  malicious.

# Creating cities
A "map" is a formatted file that represents a list of cities for which we solve a TSP. You can see an example by navigating to 
'maps/smallcity'. In general, each line follows the format <city-number/name> <x-coordinate> <y-coordinate>. Lines beginning with '#' are 
ignored. 

# Acknowledgements
[This article](https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35) by Eric Stoltz
provided significant guidance for implementation, as well as StackOverflow (as always).



