from pybench import Benchmark

regions = ['nf %d' % i for i in range(4)]


class Forms(Benchmark):

    params = [('dim', [2, 3]),
              ('degree', [1, 2, 3, 4]),
              ('qdegree', [1, 2, 3, 4]),
              ('form', ['mass', 'elasticity'])]
    method = 'forms'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = regions

if __name__ == '__main__':
    from itertools import product
    import sys
    b = Forms()
    b.combine({'FiredrakeForms_np1': 'Firedrake',
               'DolfinForms_np1': 'DOLFIN'})
    regions = map(' '.join, product(['DOLFIN', 'Firedrake'], regions))
    b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree')
    b.plot(xaxis='qdegree', regions=regions,
           xlabel='Polynomial degree (premultiplying functions)')
    if len(sys.argv) > 1:
        np = map(int, sys.argv[1:])
        b = Forms(name='DolfinFormsParallel')
        b.combine_series([('np', np)], filename='DolfinForms')
        b.save()
        b = Forms(name='FiredrakeFormsParallel')
        b.combine_series([('np', np)], filename='FiredrakeForms')
        b.save()
        b = Forms(name='FormsParallel')
        b.combine({'FiredrakeFormsParallel': 'Firedrake',
                   'DolfinFormsParallel': 'DOLFIN'})
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog')
