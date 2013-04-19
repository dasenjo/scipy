"""
fire: The FIRE optimization algorithm
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.optimize as opt


__all__ = ['fmin_fire']


def fmin_fire(func, x0, fprime=None, args=(), approx_grad=False,
              tol=1e-3, maxiter=100000, disp=None, callback=None):
    """
    Minimize a function func using the FIRE algorithm.

    Parameters
    ----------
    func : callable f(x,*args)
        Function to minimise.
    x0 : ndarray
        Initial guess.
    fprime : callable fprime(x,*args)
        The gradient of `func`.  If None, then `func` returns the function
        value and the gradient (``f, g = func(x, *args)``), unless
        `approx_grad` is True in which case `func` returns only ``f``.
    args : sequence
        Arguments to pass to `func` and `fprime`.
    approx_grad : bool
        Whether to approximate the gradient numerically (in which case
        `func` returns only the function value).
    tol : float
        The iteration stops when the norm of the gradient is smaller than tol.
    disp : int, optional
        If zero, then no output.  If a positive number, then this over-rides
        `iprint` (i.e., `iprint` gets the value of `disp`).
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    res : Results
        The optimization result represented as a ``Result`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `Result` for a description of other attributes.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'FIRE' `method` in particular.

    Notes
    -----
    The Fast Inertial Relaxation Engine is an optimization algorithm based
    on molecular dynamics with modifications to the velocity and adaptive
    time steps. The method is based on a blind skier searching for the bottom
    of a valley.

    References
    ----------
    Erik Bitzek, Pekka Koskinen, Franz Gaehler, Michael Moseler,
    and Peter Gumbsch.
    Structural Relaxation Made Simple.
    Phys. Rev. Lett. 97, 170201 (2006).

    """
    # handle fprime/approx_grad
    if approx_grad:
        fun = func
        jac = None
    elif fprime is None:
        fun = opt.MemoizeJac(func)
        jac = fun.derivative
    else:
        fun = func
        jac = fprime

    opts = {'disp': disp,
            'tol': tol,
            'maxiter': maxiter}

    res = _minimize_fire(x0, fprime=jac, fun=fun, args=args, **opts)

    return res


def _minimize_fire(x0, jac=None, fun=None, tol=1.0e-3, maxiter=100000, dt=0.1,
                   dtmax=1.0, maxmove=0.1, Nmin=5, finc=1.1, fdec=0.5,
                   astart=0.1, fa=0.99, disp=False, eps=1e-8):

    coords = np.array(x0)
    a = astart
    Nsteps = 0
    n = len(coords)
    v = np.zeros(n)

    if jac is None:
        if fun is not None:
            def fprime(x):
                return opt.approx_fprime(x, fun, eps)
        else:
            successful = False
            msg = 'No function or gradient supplied.'
            return opt.Result(nfev=0, njev=0, nit=0, message=msg, x=coords,
                              success=successful)

    else:
        fprime = jac

    for steps in range(maxiter):

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
    from scipy.optimize import rosen, rosen_der
    x = [1.3, 0.7, 0.8, 1.9, 1.2]
    res = _minimize_fire(x, jac=rosen_der)
    print('\nUsing analytical gradient.')
    print(res)
    res = _minimize_fire(x, fun=rosen)
    print('\nUsing numerical gradient.')
    print(res)
    res = _minimize_fire(x)
    print('\nNo function or gradient.')
    print(res)

if __name__ == "__main__":
    test_fire()
