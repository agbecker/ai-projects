import random, math
from copy import deepcopy

### Simple test problems (from the book)

## Boolean vector problems

# Number of ones in the vector
def max_ones(v):
    return sum(v)

# Number of leading ones
def leading_ones(v):
    count = 0
    for e in v:
        if not e:
            break
        count += 1

    return count

# Number of b-length blocks of leading ones
def leading_ones_block(v, b):
    return leading_ones(v)//b

# Boolean trap
def trap_boolean(v):
    n = len(v)
    if all(v):
        return n+2
    
    return n - sum(v)

# --------------------------------------------

### Individual definition

class Individual():
    def __init__(self, v, mutate_func):
        self.vector = v
        self.mutate_func = mutate_func
        self.fitness = -math.inf

    def mutate(self):
        self.vector = self.mutate_func(self.vector)

    def __le__(self, other):
        return self.fitness <= other.fitness
    
    def __eq__(self, other):
        return self.fitness == other.fitness
    
    def __lt__(self, other):
        return self.fitness < other.fitness


## Vector initialization

# Boolean
def random_bool_vector(n):
    return [random.randint(0,1) for _ in range(n)]

# ---------------------------------------------

### Mutations

# Random bit flip mutation
def bit_flip_mutation(v):
    p = 15
    for i, _ in enumerate(v):
        if random.randint(1,100) <= p:
            v[i] = not v[i]

    return v

# ---------------------------------------------

### Solution algorithms

## Generational Evolution Algorithms

# (mi, lambda) ES
def mi_comma_lambda(mi, lamb, time_limit = 100, feature_size = 15, eval_func = max_ones):
    population = [Individual(random_bool_vector(feature_size), bit_flip_mutation) for _ in range(lamb)]
    best = None

    for _ in range(time_limit):
        for p in population:
            fitness = eval_func(p.vector)
            p.fitness = fitness

            if best is None or fitness > eval_func(best.vector):
                best = p
        top_individuals = sorted(population)[:mi]
        population = []
        for q in top_individuals:
            for _ in range(lamb//mi):
                new = deepcopy(q)
                new.mutate()
                population.append(new)

    print(best.vector, best.fitness)


mi_comma_lambda(5,15,100,5,max_ones)