from pybench import Benchmark

regions = ['nf %d' % i for i in range(4)]


class Forms(Benchmark):

    params = [('q', [1, 2, 3, 4]),
              ('p', [1, 2, 3, 4]),
              ('form', ['mass', 'elasticity', 'poisson', 'mixed_poisson'])]
    method = 'forms'
    benchmark = 'Forms'
    plotstyle = {'total': {'marker': '*'},
                 'nf 0': {'marker': '>'},
                 'nf 1': {'marker': '<'},
                 'nf 2': {'marker': '^'},
                 'nf 3': {'marker': 'v'}}
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = regions

if __name__ == '__main__':
    b = Forms()
    b.combine_series([('variant', ['Firedrake', 'DOLFIN'])])
    b.plot(xaxis='q', regions=regions, xlabel='Polynomial degree q',
           kinds='bar,barlog', groups=['variant'],
           title='%(form)s (single core, 3D, premultiplying functions degree %(p)d)')
    b.plot(xaxis='q', regions=regions, xlabel='Polynomial degree q',
           kinds='bar', groups=['variant'], speedup=('DOLFIN',),
           ylabel='Speedup relative to DOLFIN',
           title='%(form)s (single core, 3D, premultiplying functions degree %(p)d)')
    b.plot(xaxis='p', regions=regions, kinds='bar,barlog',
           xlabel='Polynomial degree (premultiplying functions) p',
           groups=['variant'],
           title='%(form)s (single core, 3D, polynomial degree %(q)d)')
    b.plot(xaxis='p', regions=regions, kinds='bar',
           xlabel='Polynomial degree (premultiplying functions) p',
           groups=['variant'], speedup=('DOLFIN',),
           ylabel='Speedup relative to DOLFIN',
           title='%(form)s (single core, 3D, polynomial degree %(q)d)')
    b.plot(xaxis='q', regions=regions, xlabel='Polynomial degree q',
           kinds='bar,barlog', groups=['variant', 'p'],
           legend={'loc': 'best', 'ncol': 2},
           title='%(form)s (single core, 3D)')
    b.plot(xaxis='q', regions=regions, xlabel='Polynomial degree q',
           kinds='bar', groups=['variant', 'p'], speedup=('DOLFIN',),
           ylabel='Speedup relative to DOLFIN',
           legend={'loc': 'best', 'ncol': 2},
           title='%(form)s (single core, 3D)')
