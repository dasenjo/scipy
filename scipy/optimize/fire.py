"""
fire: The FIRE optimization algorithm
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import Result, approx_fprime

__all__ = ['fmin_fire']


def fmin_fire(x0, jac=None, func=None, args=(), tol=1.0e-3,
                   maxiter=100000, dt=0.1, dtmax=1.0, maxmove=0.1,
                   Nmin=5, finc=1.1, fdec=0.5, astart=0.1, fa=0.99,
                   disp=False, eps=1.e-8):
    """
    Minimize a function func using the FIRE algorithm.

    Parameters
    ----------
    x0 : ndarray
        Initial guess.
    jac : callable jac(x,*args)
        The gradient of `func`.  If None, `func` must not be none.
    func : callable f(x,*args)
        Function to minimise. If `jac` is None then it is used to approximate
        its gradient.
    args : sequence
        Arguments to pass to `func` and `jac`.
    tol : float
        The iteration stops when the norm of the gradient is smaller than tol.
    disp : int, optional
        If zero, then no output.  If a positive number, then this over-rides
        `iprint` (i.e., `iprint` gets the value of `disp`).
    maxiter : int
        Maximum number of iterations.
    epsilon : float
        Step size used when approx_grad is True, for numerically calculating
        the gradient.

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

    return _minimize_fire(x0, jac=jac, func=func, tol=tol, maxiter=maxiter,
                         dt=dt, dtmax=dtmax, maxmove=maxmove, args=args,
                         Nmin=Nmin, finc=finc, fdec=fdec,
                         astart=astart, fa=fa, disp=disp, eps=eps)


def _minimize_fire(x0, jac=None, func=None, args=(), tol=1.0e-3, maxiter=100000,
                   dt=0.1, dtmax=1.0, maxmove=0.1, Nmin=5, finc=1.1,
                   fdec=0.5, astart=0.1, fa=0.99, disp=False, eps=1.e-8):

    if jac is None:
        if func is not None:
            def fprime(x):
                return approx_fprime(x, func, eps, *args)

            def fun(x):
                return func(x, *args)
        else:
            raise ValueError('No function or gradient supplied.')
    else:
        def fprime(x):
            return jac(x, *args)

        if func is not None:
            def fun(x):
                return func(x, *args)

    coords = np.array(x0)
    a = astart
    Nsteps = 0
    n = len(coords)
    v = np.zeros(n)

    for steps in xrange(maxiter):

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
            print('Iteration %i, Gradient: %f' % (steps, f))

    if steps+1 < maxiter:
        successful = True
        msg = 'Optimization terminated successfully.'
    else:
        successful = False
        msg = 'Maximum number of iterations reached'

    res = Result(jac=grad, nfev=0, njev=steps, nit=steps,
                      message=msg, x=coords, success=successful)

    if func is not None:
        res.fun = fun(res.x)

    return res
