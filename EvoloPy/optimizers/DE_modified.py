import random
import numpy
import time
from EvoloPy.solution import solution

# -------------------------
# Reflection function
# -------------------------
def reflect(value, lower_bound, upper_bound):
    if lower_bound >= upper_bound:
        return lower_bound
    range_size = upper_bound - lower_bound
    normalized = (value - lower_bound) % (2 * range_size)
    if normalized > range_size:
        return upper_bound - (normalized - range_size)
    return lower_bound + normalized

# Differential Evolution (DE)
# mutation factor = [0.5, 2]
# crossover_ratio = [0,1]
def DE(objf, lb, ub, dim, PopSize, iters, OriginShift, Seed):
    numpy.random.seed(Seed)
    random.seed(Seed)

    mutation_factor = 0.5
    crossover_ratio = 0.7
    stopping_func = None

    # convert lb, ub to array
    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]

    # solution
    s = solution()

    s.best = float("inf")

    # initialize population
    population = []

    population_fitness = numpy.array([float("inf") for _ in range(PopSize)])

    for p in range(PopSize):
        sol = []
        for d in range(dim):
            d_val = random.uniform(lb[d], ub[d])
            sol.append(d_val)

        population.append(sol)

    population = numpy.array(population)

    # calculate fitness for all the population
    for i in range(PopSize):
        fitness = objf(population[i, :] + OriginShift)
        population_fitness[p] = fitness
        # s.func_evals += 1

        # is leader ?
        if fitness < s.best:
            s.best = fitness
            s.leader_solution = population[i, :]
    
    # Accumulated shift (just like GWO_modified)
    total_shift = numpy.zeros(dim)

    convergence_curve = numpy.zeros(iters)
    # start work
    print('DE is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    t = 0
    while t < iters:
        # should i stop
        if stopping_func is not None and stopping_func(s.best, s.leader_solution, t):
            break

        # loop through population
        for i in range(PopSize):
            # 1. Mutation

            # select 3 random solution except current solution
            ids_except_current = [_ for _ in range(PopSize) if _ != i]
            id_1, id_2, id_3 = random.sample(ids_except_current, 3)

            mutant_sol = []
            for d in range(dim):
                d_val = population[id_1, d] + mutation_factor * (
                    population[id_2, d] - population[id_3, d]
                )

                # 2. Recombination
                rn = random.uniform(0, 1)
                if rn > crossover_ratio:
                    d_val = population[i, d]

                # add dimension value to the mutant solution
                mutant_sol.append(d_val)

                mutant_sol[i, d] = reflect(
                    mutant_sol[i, d],
                    lb[d] - total_shift[d],
                    ub[d] - total_shift[d]
                )

            # 3. Replacement / Evaluation

            # clip new solution (mutant)
            #mutant_sol = numpy.clip(mutant_sol, lb, ub)

            # calc fitness
            mutant_fitness = objf(mutant_sol)
            # s.func_evals += 1

            # replace if mutant_fitness is better
            if mutant_fitness < population_fitness[i]:
                population[i, :] = mutant_sol
                population_fitness[i] = mutant_fitness

                # update leader
                if mutant_fitness < s.best:
                    shifted = True
                    s.best = mutant_fitness
                    s.leader_solution = mutant_sol
            
            if shifted:
                shift_vector = mutant_sol.copy()
                domain_size = numpy.linalg.norm(numpy.array(ub) - numpy.array(lb))

                # Threshold same as GWO_modified
                if numpy.linalg.norm(shift_vector) > 0.05 * domain_size:

                    # Accumulate shift
                    total_shift += mutant_sol

                    # Recenter positions
                    s.leader_solution -= shift_vector
                    pos -= shift_vector

                    # Reinitialize swarm: 1 at origin, rest random
                    pos[0, :] = numpy.zeros(dim)
                    for j in range(dim):
                        pos[1:, j] = (
                            numpy.random.uniform(0, 1, PopSize - 1)
                            * (ub[j] - lb[j])
                            + lb[j]
                            - total_shift[j]
                        )
            

        convergence_curve[t] = s.best
        if t % 1 == 0:
            print(
                ["At iteration " + str(t + 1) + " the best fitness is " + str(s.best)]
            )

        # increase iterations
        t = t + 1

        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergence = convergence_curve
        s.optimizer = "DEM"
        s.bestIndividual = s.leader_solution + total_shift
        s.objfname = objf.__name__

    # return solution
    return s
