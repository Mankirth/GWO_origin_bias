# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:27:50 2016

@author: Hossam Faris
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

def ESM(objf, lb, ub, dim, SearchAgents_no, Max_iter, OriginShift, Seed):
    numpy.random.seed(Seed)
    random.seed(Seed)

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # define the maximum step size
    step_size = 0.2
    # number of parents selected
    mu = SearchAgents_no / 5
    
    best_eval = float("inf")
    best = numpy.zeros(dim)
	# calculate the number of children per parent
    n_children = int((SearchAgents_no) / mu)


    # Initialize the positions of search agents at centroid
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    # Accumulated shift (just like GWO_modified)
    total_shift = numpy.zeros(dim)

    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()

    # Loop counter
    print('ES is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0, Max_iter):
        shifted = False

        for i in range(SearchAgents_no):
            # Reflection BEFORE fitness
            for j in range(dim):
                Positions[i, j] = reflect(
                    Positions[i, j],
                    lb[j] - total_shift[j],
                    ub[j] - total_shift[j]
                )

        # evaluate fitness for the population
        scores = [objf(c + OriginShift + total_shift) for c in Positions]
		# rank scores in ascending order
        ranks = numpy.argsort(numpy.argsort(scores))
		# select the indexes for the top mu ranked solutions
        selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]

        children = list()

        for i in selected:
			# check if this parent is the best solution ever seen
            if scores[i] < best_eval:
                best = Positions[i]
                best_eval = scores[i]
			# create children for parent
            for _ in range(n_children):
                child = numpy.zeros(dim)
                for j in range(dim):
                    child[j] = Positions[i][j] + ((numpy.random.uniform(0, 1) * (ub[j] - lb[j]) + lb[j]) * step_size)
                children.append(child)

		# replace population with children
        Positions = children

        if l % 1 == 0:
            print(["At iteration " + str(l) + " the best fitness is " + str(best_eval)])

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "ESM"
    s.bestIndividual = best
    s.objfname = objf.__name__

    return s
