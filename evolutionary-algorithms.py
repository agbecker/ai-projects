import random, math
from copy import deepcopy
import matplotlib.pyplot as plt

### Individual definition

class Individual():
    def __init__(self, v, mutate_func):
        self.vector = v
        self.mutate_func = mutate_func
        self.fitness = 0

    def mutate(self):
        self.vector = self.mutate_func(self.vector)

    def __le__(self, other):
        return self.fitness <= other.fitness
    
    def __eq__(self, other):
        return self.fitness == other.fitness
    
    def __lt__(self, other):
        return self.fitness < other.fitness

# --------------------------------------------

### Simple test problems (from the book)

## Boolean vector problems

# Number of ones in the vector
def max_ones(v):
    if type(v) is Individual:
        v = v.vector
    return sum(v)

# Number of leading ones
def leading_ones(v):
    if type(v) is Individual:
        v = v.vector
    count = 0
    for e in v:
        if not e:
            break
        count += 1

    return count

# Number of b-length blocks of leading ones
def leading_ones_block(v, b):
    if type(v) is Individual:
        v = v.vector
    return leading_ones(v)//b

# Boolean trap
def trap_boolean(v):
    if type(v) is Individual:
        v = v.vector
    n = len(v)
    if all(v):
        return n+2
    
    return n - sum(v)

# --------------------------------------------

## Vector initialization

# Boolean
def random_bool_vector(n):
    return [random.randint(0,1) for _ in range(n)]

# ---------------------------------------------

### Alterations to population

## Mutations

# Random bit flip mutation
def bit_flip_mutation(v):
    p = 15
    for i, _ in enumerate(v):
        if random.randint(1,100) <= p:
            v[i] = not v[i]

    return v

## Crossovers

# One point crossover
def one_point_crossover(x, y):
    u = x.vector
    v = y.vector
    n = len(u)
    point = random.randint(0,n-1)

    c1 = u[:point+1] + v[point+1:]
    c2 = v[:point+1] + u[point+1:]

    c1 = Individual(c1, x.mutate_func)
    c2 = Individual(c2, x.mutate_func)
    return c1, c2

# ---------------------------------------------

### Selection Policies

def fit_proportionate_selection(population):
    choices = []

    for p in population:
        choices += [p]*int(p.fitness)

    n = len(choices)

    return choices[random.randint(0,n-1)]


### Solution algorithms

## Generational Evolution Algorithms

# (mi, lambda) ES
def mi_comma_lambda(mi, lamb, time_limit = 100, feature_size = 15, eval_func = max_ones):
    population = [Individual(random_bool_vector(feature_size), bit_flip_mutation) for _ in range(lamb)]
    best = None

    x = []
    y = []

    for i in range(time_limit):
        x.append(i)
        for p in population:
            fitness = eval_func(p.vector)
            p.fitness = fitness

            if best is None or fitness > eval_func(best.vector):
                best = p
        y.append(sum([j.fitness for j in population])/len(population))

        top_individuals = sorted(population)[:mi]
        population = []
        for q in top_individuals:
            for _ in range(lamb//mi):
                new = deepcopy(q)
                new.mutate()
                population.append(new)

    print(best.vector, best.fitness)

    plt.plot(x,y)
    plt.show()

# Genetic Algorithm
def genetic_algorithm(popsize, feature_size = 15, time_limit = 100, eval_func = trap_boolean, crossover = one_point_crossover, select = fit_proportionate_selection):
    if popsize%2:
        popsize += 1

    population = [Individual(random_bool_vector(feature_size), bit_flip_mutation) for _ in range(popsize)]
    best = None

    y = []
    yb = []
    ybg = []

    for _ in range(time_limit):

        for p in population:
            fitness = eval_func(p)
            p.fitness = fitness
            
            if best is None or fitness > best.fitness:
                best = p

            top_pop = []

        y.append(sum([j.fitness for j in population])/len(population))
        yb.append(best.fitness)
        ybg.append(max([j.fitness for j in population]))

        for _ in range(popsize//2):
            pa = select(population)
            pb = select(population)
            c1, c2 = crossover(deepcopy(pa), deepcopy(pb))
            c1.mutate()
            c2.mutate()
            top_pop.extend([c1, c2])
        population = top_pop

    print(list(map(lambda x: int(x), best.vector)), best.fitness)

    plt.plot(y)
    plt.plot(yb)
    plt.plot(ybg)
    plt.show()

# ========== Main =============

genetic_algorithm(15, feature_size=8, time_limit=100)