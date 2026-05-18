# -*- coding: utf-8 -*-
"""
PSO with origin-shifting, restart, and reflection mechanism.
Mirrors GWO_modified behavior.
"""

import numpy as np
import random
from EvoloPy.solution import solution
import time


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


# -------------------------
# Modified PSO
# -------------------------
def PSOM(objf, lb, ub, dim, PopSize, iters, OriginShift, Seed):

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

    # Accumulated shift (just like GWO_modified)
    total_shift = np.zeros(dim)

    convergence_curve = np.zeros(iters)

    # --------------------------------------
    print('PSO_modified is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # --------------------------------------
    # PSO main loop
    # --------------------------------------
    for l in range(iters):
        shifted = False

        # --------------------------------------
        # Evaluate particles
        # --------------------------------------
        for i in range(PopSize):
            # Reflection BEFORE fitness
            for j in range(dim):
                pos[i, j] = reflect(
                    pos[i, j],
                    lb[j] - total_shift[j],
                    ub[j] - total_shift[j]
                )

            # Evaluate in shifted space
            fitness = objf(pos[i, :] + OriginShift + total_shift)

            # Update personal best
            if fitness < pBestScore[i]:
                pBestScore[i] = fitness
                pBest[i, :] = pos[i, :].copy()

            # Update global best
            if fitness < gBestScore:
                shifted = True
                gBestScore = fitness
                gBest = pos[i, :].copy()

        # --------------------------------------
        # Shift and reset condition (RESTART)
        # --------------------------------------
        if shifted:
            shift_vector = gBest.copy()
            domain_size = np.linalg.norm(np.array(ub) - np.array(lb))

            # Threshold same as GWO_modified
            if np.linalg.norm(shift_vector) > 0.05 * domain_size:

                # Accumulate shift
                total_shift += gBest

                # Recenter positions
                gBest -= shift_vector
                pBest -= shift_vector
                pos -= shift_vector

                # Reinitialize swarm: 1 at origin, rest random
                pos[0, :] = np.zeros(dim)
                for j in range(dim):
                    pos[1:, j] = (
                        np.random.uniform(0, 1, PopSize - 1)
                        * (ub[j] - lb[j])
                        + lb[j]
                        - total_shift[j]
                    )

                # Reset velocities too
                vel = np.zeros((PopSize, dim))

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
    s.optimizer = "PSOM"
    s.bestIndividual = gBest + total_shift
    s.objfname = objf.__name__
    s.shift = total_shift

    return s
