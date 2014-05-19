from pybench import Benchmark

regions = ['nf %d' % i for i in range(3)]


class MixedForms(Benchmark):

    params = [('dim', [2, 3]),
              ('degree', [1, 2, 3, 4]),
              ('qdegree', [1, 2, 3, 4]),
              ('form', ['poisson'])]
    method = 'mixed_forms'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = regions

if __name__ == '__main__':
    from itertools import product
    import sys
    b = MixedForms()
    b.combine({'FiredrakeMixedForms_np1': 'Firedrake',
               'DolfinMixedForms_np1': 'DOLFIN'})
    regions = map(' '.join, product(['DOLFIN', 'Firedrake'], regions))
    b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree',
           kinds='bar', legend='best')
    b.plot(xaxis='qdegree', regions=regions, kinds='bar', legend='best',
           xlabel='Polynomial degree (premultiplying functions)')
    if len(sys.argv) > 1:
        np = map(int, sys.argv[1:])
        b = MixedForms(name='DolfinMixedFormsParallel')
        b.combine_series([('np', np)], filename='DolfinMixedForms')
        b.save()
        b = MixedForms(name='FiredrakeMixedFormsParallel')
        b.combine_series([('np', np)], filename='FiredrakeMixedForms')
        b.save()
        b = MixedForms(name='MixedFormsParallel')
        b.combine({'FiredrakeMixedFormsParallel': 'Firedrake',
                   'DolfinMixedFormsParallel': 'DOLFIN'})
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog')
