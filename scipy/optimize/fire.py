"""
fire: The FIRE optimization algorithm
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.optimize

__all__ = ['fire']


def _minimize_fire(x0, fprime, tol=1.0e-3, nsteps=100000, dt=0.1, dtmax=1.0,
                   maxstep=0.5, Nmin=5, finc=1.1, fdec=0.5, astart=0.1,
                   fa=0.99, disp=False):

    coords = x0.copy()
    a = astart
    steps = 0
    Nsteps = 0
    v = np.zeros(len(coords))

    for k in range(nsteps):
        grad = fprime(coords)
        f = np.sqrt(np.vdot(grad, grad))
        if f < tol:
            break
        P = np.vdot(-grad, v)
        if P > 0.0:
            v = (1.0 - a) * v + a * (-grad / f) * np.sqrt(np.vdot(v, v))
            if (Nsteps > Nmin):
                dt = min(dt * finc, dtmax)
                a *= fa
                Nsteps += 1
            else:
                v = np.zeros(2 * npart)
                a = astart
                dt = dt * fdec
                Nsteps = 0
        v -= dt * grad
        dr = dt * v
        normdr = np.sqrt(np.vdot(dr, dr))
        if normdr > maxmove:
            dr *= maxmove / normdr
        coords += dr
        steps += 1

    return Result(jac=grad, nfev=0, njev=steps, nit=steps,
                  status=warnflag, message=task_str, x=coords,
                  success=(warnflag == 0))
