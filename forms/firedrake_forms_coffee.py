from firedrake_forms import FiredrakeForms
from firedrake import *
from pyop2.ir.ast_plan import V_OP_UAJ


class FiredrakeFormsCoffee(FiredrakeForms):

    params = [('degree', [1, 2, 3, 4]),
              ('qdegree', [1, 2, 3, 4]),
              ('form', ['mass', 'elasticity', 'poisson', 'mixed_poisson']),
              ('licm', [False, True]),
              ('vect', [(False, None), (True, (V_OP_UAJ, 1))])]

    def forms(self, degree=1, qdegree=1, dim=3, form='mass', licm=False, vect=(False, None)):
        parameters["coffee"]["licm"] = licm
        parameters["coffee"]["ap"] = vect[0]
        parameters["coffee"]["vect"] = vect[1]
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
    b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree')
    b.plot(xaxis='qdegree', regions=regions,
           xlabel='Polynomial degree (premultiplying functions)')
