#Particle Swarm Optimization implementation 
import numpy as np
from Rastrigin import rastrigin

def particle_swarm(population_size=50, generations=100, w=0.7, c1=1.5, c2=1.5, bounds=(-5.12, 5.12)):
    dim = 2
    lb, ub = bounds
    
    position = np.random.uniform(lb, ub, (population_size, dim)) #randomply positioning each particle 
    velocity = np.random.uniform(-1, 1, (population_size, dim)) #random small velovity in direction of X and Y

    p_best = position.copy() #Each particle remembers its own best position
    p_best_fitness = np.array([rastrigin(p) for p in position]) #how good the position was

    g_best_idx = np.argmin(p_best_fitness)
    g_best = p_best[g_best_idx] #the position of the global best so far
    g_best_fitness = p_best_fitness[g_best_idx] #the value of global best so far

    best_fitness_per_gen = []

    for gen in range(generations):
        for i in range(population_size):

            r1 = np.random.rand(dim)#adding randomness to how particles are influenced
            r2 = np.random.rand(dim)#adding randomness to how particles are influenced

            cognitive = c1 * r1 * (p_best[i] - position[i]) #how much a particle is pulled toward its own best position
            social = c2 * r2 * (g_best - position[i]) #how much a particle is pulled toward global best position
            velocity[i] = w * velocity[i] + cognitive + social # updated velocity for pulling toward the own and global best

            position[i] += velocity[i] #updating the poistion based on new velocity
            position[i] = np.clip(position[i], lb, ub) #clamping to ensure we saty inside the boundaries

            fitness = rastrigin(position[i]) #evaluating current fitness and situation
            if fitness < p_best_fitness[i]:
                p_best[i] = position[i]
                p_best_fitness[i] = fitness
                #If the new position is better than the particle’s personal best, update it
                if fitness < g_best_fitness:
                    g_best = position[i]
                    g_best_fitness = fitness
                    #If the new position is also better than the global best, update that too
        best_fitness_per_gen.append(g_best_fitness)

    return g_best, g_best_fitness, best_fitness_per_gen
