from firedrake_forms import FiredrakeForms
from firedrake import *
from pyop2.coffee.ast_plan import V_OP_UAJ


class FiredrakeFormsCoffee(FiredrakeForms):

    params = [('degree', [1, 2, 3, 4]),
              ('qdegree', [1, 2, 3, 4]),
              ('form', ['mass', 'elasticity', 'poisson', 'mixed_poisson']),
              ('opt', [(False, False, None), (True, False, None), (True, True, None)]
               + [(True, True, (V_OP_UAJ, i)) for i in range(1, 5)])]

    def forms(self, degree=1, qdegree=1, dim=3, form='mass', opt=(False, False, None)):
        parameters["coffee"]["licm"] = opt[0]
        parameters["coffee"]["ap"] = opt[2]
        parameters["coffee"]["vect"] = opt[2]
        super(FiredrakeFormsCoffee, self).forms(degree, qdegree, dim, form)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    # Benchmark
    b = FiredrakeFormsCoffee()
    b.main(benchmark=True, save=None)

    # Plot
    regions = ['nf %d' % i for i in range(4)]
    b.plot(xaxis='opt', regions=regions, kinds='bar', legend='best',
           xlabel='COFFEE Optimisations (LICM, AP, VECT)',
           xvalues=['n/n/n', 'y/n/n', 'y/y/n'] + ['y/y/(4, %d)' % i for i in range(1, 5)])
