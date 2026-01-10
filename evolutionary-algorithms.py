import random
import math
from copy import deepcopy

# Problema toy: 
# Vetores de 10 valores

def fitness(v):
    [a, b, c, d, e,
     f, g, h, i, j] = v
    
    return a + b -c*d + e*f +(g-h)*(i-j)

def init_vector():
    return [random.randrange(-100, 100) for _ in range(10)]

def random_population(n):
    return [init_vector() for _ in range(15)]

def crossover(a, b):
    n = len(a)

    new = []

    for i in range(n):
        from_a = random.randint(0,1)

        if from_a:
            new.append(a[i])
        else:
            new.append(b[i])

    return new

def show_population_info(pop):
    n = len(pop)

    fits = []
    for i in range(n):
        fit = fitness(pop[i])
        print(f'[{i+1}] | {pop[i]} = {fit}')
        fits.append(fit)
    print('---------------------')
    print('Best:', max(fits))
    print('Average', sum(fits)/n)
    print('=====================\n')

#-----------------------

# Differential evolution

def differential_evolution(num_iters, popsize = 15, mutation_rate = 0.5):
    population = random_population(popsize)
    parents = None
    best = None
    best_fit = -math.inf

    for k in range(num_iters):
        print(f'Epoch {k+1}')
        show_population_info(population)

        for i, p in enumerate(population):
            fit = fitness(p)
            if parents is not None and fitness(parents[i]) > fit:
                population[i] = parents[i]
            if best is None or fit > best_fit:
                best = p
                best_fit = fit

        parents = population

        for i, q in enumerate(parents):
            a = q
            while a == q:
                aa = random.choice(parents)
                a = deepcopy(aa)
            
            b = a
            while b == q or b == a:
                bb = random.choice(parents)
                b = deepcopy(bb)

            c = b
            while c == q or c == a or c == b:
                cc = random.choice(parents)
                c = deepcopy(cc)
            
            bminusc = [mutation_rate*(b[i] - c[i]) for i in range(len(b))]
            d = [a[i] + bminusc[i] for i in range(len(a))]
            population[i] = crossover(d, deepcopy(q))

    print(f'Best individual: {best} = {best_fit}')


if __name__ == '__main__':
    differential_evolution(20)