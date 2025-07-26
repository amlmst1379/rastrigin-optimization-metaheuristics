#Genteic algorithm implementation
import numpy as np
from utils.rastrigin import rastrigin


def genetic_algorithm(
    #At first, we will generate our population 
    pop_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1, bounds=(-5.12, 5.12)
):
    #Our analysis is for two dimensions
    dim = 2  # 2D: x and y
    lb, ub = bounds

    # Initialize population
    population = np.random.uniform(lb, ub, (pop_size, dim))

    # Evaluate initial fitness
    fitness = np.array([rastrigin(ind) for ind in population])

    best_fitness_per_gen = [] #To store the best fitness value per generation
    best_solution = population[np.argmin(fitness)] #best solution that contains the best value for fitness
    best_fitness = np.min(fitness) #The minimum fitness per generation

    # Evolution loop
    for gen in range(generations):
        #New population from the existing parents population to be filled
        new_population = []

        for _ in range(pop_size // 2):  # Each iteration creates 2 children
            # Selection of parents (better performing guesses) for the generated children 
            parents = selection(population, fitness, k=3)

            # Combining the best parents to generate the best child
            child1, child2 = crossover(parents[0], parents[1], crossover_rate)

            # Mutation to help explore new possibilities and avoid getting stuck.
            child1 = mutate(child1, mutation_rate, lb, ub)
            child2 = mutate(child2, mutation_rate, lb, ub)

            new_population.extend([child1, child2])

        # Evaluating new population
        population = np.array(new_population)
        fitness = np.array([rastrigin(ind) for ind in population])

        # Saving best result of this generation
        gen_best_idx = np.argmin(fitness)
        gen_best_fit = fitness[gen_best_idx]

        if gen_best_fit < best_fitness:
            best_fitness = gen_best_fit
            best_solution = population[gen_best_idx]
            #If the children of the best parents introduce a better fitness, save them as the best result for now
        best_fitness_per_gen.append(best_fitness)

    return best_solution, best_fitness, best_fitness_per_gen

def selection(population, fitness, k=3):
    #Selecting two best available parents
    selected = np.random.choice(len(population), k)
    best_idx = selected[np.argmin(fitness[selected])]
    second_idx = selected[np.argsort(fitness[selected])[1]]
    return population[best_idx], population[second_idx]


def crossover(parent1, parent2, rate):
    if np.random.rand() < rate:
        alpha = np.random.rand()
        child1 = alpha * parent1 + (1 - alpha) * parent2 #randomly combining the best existing parents
        child2 = alpha * parent2 + (1 - alpha) * parent1 #randomly combining the best existing parents
        return child1, child2
    else:
        return parent1.copy(), parent2.copy()


def mutate(individual, rate, lb, ub):
    for i in range(len(individual)):
        if np.random.rand() < rate:
            individual[i] += np.random.normal(0, 0.1) #tweaking the current children to avoid getting stuck
            individual[i] = np.clip(individual[i], lb, ub)
    return individual
