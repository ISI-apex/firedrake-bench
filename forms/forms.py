from pybench import Benchmark

regions = ['nf %d' % i for i in range(4)]


class Forms(Benchmark):

    params = [('p', [1, 2, 3, 4]),
              ('q', [1, 2, 3, 4]),
              ('form', ['mass', 'elasticity', 'poisson', 'mixed_poisson'])]
    method = 'forms'
    name = 'Forms'
    plotstyle = {'total': {'marker': '*'},
                 'nf 0': {'marker': '>'},
                 'nf 1': {'marker': '<'},
                 'nf 2': {'marker': '^'},
                 'nf 3': {'marker': 'v'}}
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = regions

if __name__ == '__main__':
    import sys
    b = Forms()
    b.combine_series([('np', [1]), ('variant', ['Firedrake', 'DOLFIN'])])
    b.plot(xaxis='p', regions=regions, xlabel='Polynomial degree',
           kinds='bar,barlog', legend='best', groups=['variant'])
    b.plot(xaxis='p', regions=regions, xlabel='Polynomial degree',
           kinds='bar', legend='best', groups=['variant'], speedup=('DOLFIN',),
           ylabel='Speedup relative to DOLFIN')
    b.plot(xaxis='q', regions=regions, kinds='bar,barlog', legend='best',
           xlabel='Polynomial degree (premultiplying functions)',
           groups=['variant'])
    b.plot(xaxis='q', regions=regions, kinds='bar', legend='best',
           xlabel='Polynomial degree (premultiplying functions)',
           groups=['variant'], speedup=('DOLFIN',),
           ylabel='Speedup relative to DOLFIN')
    if len(sys.argv) > 1:
        np = map(int, sys.argv[1:])
        b = Forms(name='FormsParallel')
        b.combine_series([('np', np), ('variant', ['Firedrake', 'DOLFIN'])],
                         filename='Forms')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'])
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
               ylabel='Speedup relative to DOLFIN on 1 core')
