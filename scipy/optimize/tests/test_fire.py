"""
Unit tests for the FIRE optimization algorithm.
"""
from __future__ import division, print_function, absolute_import

def test_fire():
    import numpy as np
    from scipy.optimize import fmin_fire, minimize
    from beale import Beale as func
    f = func()

    def grad(coords):
        return f.getEnergyGradient(coords)[1]

    def fun(coords):
        return f.getEnergyGradient(coords)[0]

    x = [np.random.uniform(f.xmin[0], f.xmax[0]), np.random.uniform(f.xmin[1], f.xmax[1])]
    print('\nStarting point x=', x)
    res = fmin_fire(x, jac=grad, func=fun)
    print('\nFIRE')
    print(res)
    res = minimize(f.getEnergy, x, method='BFGS',
                       options={'gtol': 1e-2})
    print('\nBFGS')
    print(res)
    res = minimize(fun, x, method='fire', jac=grad)
    print('\n Using FIRE from minimize')
    print(res,'\n')

if __name__ == "__main__":
    test_fire()
