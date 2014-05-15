import numpy as np


def compute_spectrum(mat, filename):
    from slepc4py import SLEPc
    eps = SLEPc.EPS().create()
    eps.setDimensions(10)
    eps.setOperators(mat)
    # Compute 10 eigenvalues with smallest magnitude
    eps.setWhichEigenpairs(eps.Which.SMALLEST_MAGNITUDE)
    eps.solve()
    smallest = np.array([eps.getEigenvalue(i) for i in range(eps.getConverged())])
    np.save(filename + '_smallest.npy', smallest)
    # Compute 10 eigenvalues with largest magnitude
    eps.setWhichEigenpairs(eps.Which.LARGEST_MAGNITUDE)
    eps.solve()
    largest = np.array([eps.getEigenvalue(i) for i in range(eps.getConverged())])
    np.save(filename + '_largest.npy', largest)
