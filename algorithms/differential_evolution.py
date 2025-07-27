import numpy as np
from Rastrigin import rastrigin

def differential_evolution(population_size=50, generations=100, F=0.8, CR=0.9, bounds=(-5.12, 5.12)):
    dim = 2 #Since we have X and Y
    lb, ub = bounds #lower and upper bounds

    population = np.random.uniform(lb, ub, (population_size, dim)) #generating random population
    fitness = np.array([rastrigin(ind) for ind in population]) #Fitness values for the random guesses

    best_solution = population[np.argmin(fitness)] #To store the best solution per generation
    best_fitness = np.min(fitness) #best overall fitness 
    best_fitness_per_gen = [] #best fitness in each generation
    #Evaluation 
    for gen in range(generations): 
        for i in range(population_size): #looping for each individual in population
            
            idxs = [idx for idx in range(population_size) if idx != i] # random individuals other than the current one
            a, b, c = population[np.random.choice(idxs, 3, replace=False)] #3 random choices from the current population 
            mutant = np.clip(a + F * (b - c), lb, ub) #mutatting based on the paths other have taken. 

            crossover = np.random.rand(dim) < CR
            if not np.any(crossover):
                crossover[np.random.randint(0, dim)] = True #keeping from the current parent
            trial = np.where(crossover, mutant, population[i]) #The offspring 

            f_trial = rastrigin(trial) #evaluating the mutant gene
            if f_trial < fitness[i]: #if better than current individual replace it
                population[i] = trial
                fitness[i] = f_trial

                if f_trial < best_fitness: #updatring the global best
                    best_fitness = f_trial
                    best_solution = trial

        best_fitness_per_gen.append(best_fitness)

    return best_solution, best_fitness, best_fitness_per_gen

