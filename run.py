#!/usr/bin/env python

from argparse import ArgumentParser
from datetime import datetime
from subprocess import check_call


def run(benchmark, np, socket=0, core=0):
    for n in np:
        cmd = ['mpirun']
        for p in range(core, n + core):
            cmd += ['-np', '1', 'likwid-pin', '-c', 'S%d:%d' % (socket, p)]
            cmd += ['python', benchmark, ':']
        print datetime.now(), benchmark, 'started'
        print ' '.join(cmd)
        check_call(cmd)
        print datetime.now(), benchmark, 'finished'
        core += n

if __name__ == '__main__':
    p = ArgumentParser(description="Run a benchmark")
    p.add_argument('--socket', '-s', type=int, default=0, help="socket to pin to")
    p.add_argument('--core', '-c', type=int, default=0, help="inital core to pin to")
    p.add_argument('benchmark', help="benchmark to run")
    p.add_argument('np', type=int, nargs='+',
                   help="number of processes to run")
    run(**vars(p.parse_args()))
