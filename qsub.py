#!/usr/bin/env python

from argparse import ArgumentParser
from subprocess import check_call
from tempfile import NamedTemporaryFile

pbs = """\
#PBS -N %(jobname)s
#PBS -l walltime=8:00:00
#PBS -l select=%(nodes)d:ncpus=%(cpus)d:mem=15gb:sandyb=true
#PBS -l place=excl
#PBS -q %(queue)s
#PBS -m eba
#PBS -M %(email)s

echo Running in $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

LOGFILE=%(jobname)s.${PBS_JOBID}.log

%(env)s

echo -n Started at
date

echo
echo Running %(script)s
echo

mpiexec python %(script)s.py %(args)s 2>&1  | tee $LOGFILE

echo -n Finished at
date
"""


def run(benchmark, template, nodes, queue, email, env, args, np):
    """Submit a batch job

    :param benchmark: benchmark to run
    :param template: template to create PBS script from
    :param nodes: number of nodes to run for
    :param queue: queue to submit to
    :param email: email address to notifiy
    :param env: env file to source
    :param args: list of additional argument to pass to benchmark script
    :param np: number of processes to run for
    """
    d = {'script': benchmark,
         'nodes': nodes,
         'queue': queue,
         'email': email,
         'env': open(env).read(),
         'args': ' '.join(args)}
    if template:
        with open(template) as f:
            template = f.read()
    for n in np:
        d['cpus'] = n
        d['jobname'] = '%s_%02d' % (benchmark[:12], n)
        with NamedTemporaryFile(prefix=d['jobname']) as f:
            f.write((template or pbs) % d)
            f.file.flush()
            check_call(['qsub', f.name])

if __name__ == '__main__':
    p = ArgumentParser(description="Submit a batch job")
    p.add_argument('--template', '-t', help="template to create PBS script from")
    p.add_argument('--queue', '-q', help="queue to submit to")
    p.add_argument('--nodes', '-n', type=int, help="number of nodes")
    p.add_argument('--email', '-m',
                   help="email address to send status messages to")
    p.add_argument('--env', '-e', help="environment script to source")
    p.add_argument('--np', type=int, nargs='+',
                   help="number of processes per node")
    p.add_argument('benchmark', help="benchmark to run")
    p.add_argument('args', help="arguments to pass to benchmarks script",
                   nargs="*")
    run(**vars(p.parse_args()))
