from firedrake_forms import FiredrakeForms
from firedrake import *
from pyop2.coffee.ast_plan import V_OP_UAJ


class FiredrakeFormsCoffee(FiredrakeForms):
    name = 'FiredrakeFormsCoffee'
    series = {}
    params = [('q', [1, 2, 3, 4]),
              ('p', [1, 2, 3, 4]),
              ('form', ['mass', 'elasticity', 'poisson', 'mixed_poisson']),
              ('opt', [(False, False, None), (True, False, None), (True, True, None)]
               + [(True, True, (V_OP_UAJ, i)) for i in range(1, 5)])]

    def forms(self, q=1, p=1, dim=3, max_nf=3, form='mass', opt=(False, False, None)):
        parameters["coffee"]["licm"] = opt[0]
        parameters["coffee"]["ap"] = opt[1]
        parameters["coffee"]["vect"] = opt[2]
        super(FiredrakeFormsCoffee, self).forms(q, p, dim, max_nf, form)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    # Benchmark
    b = FiredrakeFormsCoffee()
    b.main(load=None)

    # Plot
    regions = ['nf %d' % i for i in range(4)]
    b.plot(xaxis='opt', regions=regions, kinds='bar,barlog',
           xlabel='COFFEE Optimisations (LICM, AP, VECT)',
           xvalues=['n/n/n', 'y/n/n', 'y/y/n'] + ['y/y/(4, %d)' % i for i in range(1, 5)],
           title='%(form)s (single core, 3D, degree q = %(q)d, premultiplying degree p = %(p)d)')
    b.plot(xaxis='opt', regions=regions, kinds='bar',
           xlabel='COFFEE Optimisations (LICM, AP, VECT)',
           ylabel='Speedup over unoptimised baseline', speedup=((False, False, None),),
           xvalues=['n/n/n', 'y/n/n', 'y/y/n'] + ['y/y/(4, %d)' % i for i in range(1, 5)],
           title='%(form)s (single core, 3D, degree q = %(q)d, premultiplying degree p = %(p)d)')
    for i, r in enumerate(regions):
        b.plot(xaxis='opt', regions=[r], kinds='bar,barlog',
               xlabel='COFFEE Optimisations (LICM, AP, VECT)', groups=['form'],
               figname='FiredrakeFormsCoffee_nf%d' % i,
               xvalues=['n/n/n', 'y/n/n', 'y/y/n'] + ['y/y/(4, %d)' % i for i in range(1, 5)],
               title=str(i) + ' premultiplying functions (single core, 3D, degree q = %(q)d, premultiplying degree p = %(p)d)')
