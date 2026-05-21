# -*- coding: utf-8 -*-
"""
PSO with origin-shifting, restart, and reflection mechanism.
Mirrors GWO_modified behavior.
"""

import numpy as np
import random
from EvoloPy.solution import solution
import time

def PSO(objf, lb, ub, dim, PopSize, iters, OriginShift, Seed):

    # Seed RNG
    np.random.seed(Seed)
    random.seed(Seed)

    # PSO parameters
    Vmax = 6
    wMax = 0.9
    wMin = 0.2
    c1 = 2
    c2 = 2

    s = solution()

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # --------------------------------------
    # Initial population and bookkeeping
    # --------------------------------------
    vel = np.zeros((PopSize, dim))

    pBestScore = np.full(PopSize, float("inf"))
    pBest = np.zeros((PopSize, dim))

    gBestScore = float("inf")
    gBest = np.zeros(dim)

    pos = np.zeros((PopSize, dim))
    for j in range(dim):
        pos[:, j] = np.random.uniform(0, 1, PopSize) * (ub[j] - lb[j]) + lb[j]

    convergence_curve = np.zeros(iters)

    # --------------------------------------
    print('PSO_modified is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # --------------------------------------
    # PSO main loop
    # --------------------------------------
    for l in range(iters):

        # --------------------------------------
        # Evaluate particles
        # --------------------------------------
        for i in range(PopSize):
            pos[i] = np.clip(pos[i], lb, ub)

            # Evaluate
            fitness = objf(pos[i, :] + OriginShift)

            # Update personal best
            if fitness < pBestScore[i]:
                pBestScore[i] = fitness
                pBest[i, :] = pos[i, :].copy()

            # Update global best
            if fitness < gBestScore:
                gBestScore = fitness
                gBest = pos[i, :].copy()

        # --------------------------------------
        # Velocity and position update
        # --------------------------------------
        w = wMax - l * ((wMax - wMin) / iters)

        for i in range(PopSize):
            for j in range(dim):
                r1 = random.random()
                r2 = random.random()
                vel[i, j] = (
                    w * vel[i, j]
                    + c1 * r1 * (pBest[i, j] - pos[i, j])
                    + c2 * r2 * (gBest[j] - pos[i, j])
                )

                # Clamp velocity
                vel[i, j] = np.clip(vel[i, j], -Vmax, Vmax)

                pos[i, j] = pos[i, j] + vel[i, j]

        convergence_curve[l] = gBestScore

        if l % 1 == 0:
            print(f"Iter {l} | Best = {gBestScore}")

    # --------------------------------------
    timerEnd = time.time()

    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "PSO"
    s.bestIndividual = gBest
    s.objfname = objf.__name__

    return s
