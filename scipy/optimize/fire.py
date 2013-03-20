"""
fire: The FIRE optimization algorithm
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.optimize as opt

__all__ = ['_minimize_fire']


def _minimize_fire(x0, fprime, tol=1.0e-3, maxiter=100000, dt=0.1, dtmax=1.0,
                   maxmove=0.1, Nmin=5, finc=1.1, fdec=0.5, astart=0.1,
                   fa=0.99, disp=False):

    coords = np.array(x0)
    a = astart
    steps = 0
    Nsteps = 0
    n = len(coords)
    v = np.zeros(n)

    for k in range(maxiter):

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
            v = np.zeros(n)
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

        if disp:
            print("some stuff")

    if steps < maxiter:
        successful = True
        msg = 'Optimization terminated successfully.'
    else:
        successful = False
        msg = 'Maximum number of iterations has been exceeded'

    return opt.Result(jac=grad, nfev=0, njev=steps, nit=steps,
                      message=msg, x=coords, success=successful)


def test_fire():
    from scipy.optimize import rosen_der
    x = [1.3, 0.7, 0.8, 1.9, 1.2]
    _minimize_fire(x, rosen_der)

if __name__ == "__main__":
    test_fire()
