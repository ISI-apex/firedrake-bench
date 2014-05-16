from firedrake_assembly import FiredrakeAssembly
from firedrake import *
from pyop2.ir.ast_plan import V_OP_UAJ


class FiredrakeAssemblyCoffee(FiredrakeAssembly):

    params = [('size', [16]),
              ('dim', [2, 3]),
              ('degree', [1, 2, 3, 4]),
              ('licm', [False, True]),
              ('vect', [(False, None), (True, (V_OP_UAJ, 1))])]

    def assembly(self, size=32, degree=1, dim=2, fs='scalar', licm=False, vect=(False, None)):
        parameters["coffee"]["licm"] = licm
        parameters["coffee"]["ap"] = vect[0]
        parameters["coffee"]["vect"] = vect[1]
        super(FiredrakeAssemblyCoffee, self).assembly(size, degree, dim, fs)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    # Benchmark
    b = FiredrakeAssemblyCoffee()
    b.main(benchmark=True, save=None)

    # Plot
    from itertools import product
    r0 = ['mass', 'laplace', 'helmholtz']
    r1 = ['0', '1', '2', '3']

    # Create separate plots for mass, laplace, helmholtz
    for r in r0:
        regions = map(' '.join, product([r], ['premult'], r1))
        b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree',
               figname=b.name + '_' + r, wscale=0.7)
