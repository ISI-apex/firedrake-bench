#!/usr/bin/env python

from argparse import ArgumentParser
from datetime import datetime
from subprocess import check_call


def run(benchmark, np, socket=None, core=0):
    for n in np:
        cmd = ['mpirun']
        for p in range(core, n + core):
            #pin = 'N:%d' % p if socket is None else 'S%d:%d' % (socket, p)
            #cmd += ['-np', '1', 'likwid-pin', '-c', pin, 'python'] + benchmark
            cmd += ['-np', '1', 'python'] + benchmark
            #cmd += [':']
        print(datetime.now(), benchmark, 'started')
        print(' '.join(cmd))
        check_call(cmd)
        print(datetime.now(), benchmark, 'finished')

if __name__ == '__main__':
    p = ArgumentParser(description="Run a benchmark pinned to CPU cores")
    p.add_argument('--socket', '-s', type=int, help="socket to pin to")
    p.add_argument('--core', '-c', type=int, default=0,
                   help="inital core to pin to")
    p.add_argument('--np', type=int, nargs='+', default=[1],
                   help="number of processes to run")
    p.add_argument('benchmark', help="benchmark to run", nargs='+')
    run(**vars(p.parse_args()))
