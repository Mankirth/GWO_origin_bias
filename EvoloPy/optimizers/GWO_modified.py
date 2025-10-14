# -*- coding: utf-8 -*-
"""
Modified GWO algorithm with origin-shifting, restart and reflect mechanism
"""

import random
import numpy
import math
from EvoloPy.solution import solution
import time

def reflect(value, lower_bound, upper_bound):

    if lower_bound >= upper_bound:
        return lower_bound  
    
    range_size = upper_bound - lower_bound
    
    normalized = (value - lower_bound) % (2 * range_size)
    
    if normalized > range_size:
        return upper_bound - (normalized - range_size)
    return lower_bound + normalized

def GWO_modified(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    # Initialize alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize positions
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]

    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()

    print('GWO_modified is optimizing "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    total_shift = [0] * dim

    for l in range(0, Max_iter):
        # Track if shift occurred in this iteration
        shifted = False
        
        for i in range(0, SearchAgents_no):
            # Apply reflection with current shift
            for j in range(dim):
                Positions[i, j] = reflect(Positions[i, j], lb[j] - total_shift[j], ub[j] - total_shift[j])

            # Evaluate in original space
            fitness = objf(Positions[i, :] + total_shift)

            # Update leaders
            if fitness < Alpha_score:
                shifted = True  # Mark that we'll need to shift
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()

            elif fitness > Alpha_score and fitness < Beta_score:
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()

            elif fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()

        # Perform shift and reinitialization if new alpha found
        if shifted:
            shift_vector = Alpha_pos.copy()
            # Update total shift
            if numpy.linalg.norm(shift_vector) > 0.05 * numpy.linalg.norm(numpy.array(ub)-numpy.array(lb)):
                total_shift += Alpha_pos
                Alpha_pos -= shift_vector
                Beta_pos -= shift_vector
                Delta_pos -= shift_vector
                ub += shift_vector
                lb += shift_vector
            
                # Reinitialize all positions randomly (except alpha)
                for i in range(dim):
                    if SearchAgents_no > 1:  # Only if we have multiple agents
                        # Keep alpha position at origin
                        Positions[0, :] = numpy.zeros(dim)  # Alpha stays at origin
                        # Randomize other positions
                        Positions[1:, i] = numpy.random.uniform(0, 1, SearchAgents_no-1) * (ub[i] - lb[i]) + lb[i] - total_shift[i]


        # Standard GWO update
        a = 2 - l * (2 / Max_iter)

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1, r2 = random.random(), random.random()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1, r2 = random.random(), random.random()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1, r2 = random.random(), random.random()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                Positions[i, j] = (X1 + X2 + X3) / 3

        Convergence_curve[l] = Alpha_score

        if l % 1 == 0:
            print(f"Iteration {l}: best fitness = {Alpha_score}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "GWO_modified"
    s.bestIndividual = Alpha_pos + total_shift
    s.objfname = objf.__name__

    return s