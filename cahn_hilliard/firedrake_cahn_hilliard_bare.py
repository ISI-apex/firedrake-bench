import os
import sys
import time
import argparse
import resource
import platform
import numpy as np
from collections import OrderedDict

import timings

from mpi4py import MPI
from firedrake import *

from firedrake_cahn_hilliard_problem import CahnHilliardProblem

def size_from_string(s):
    s = s.strip()
    if s.endswith('G'):
        p = 3
    elif s.endswith('M'):
        p = 2
    elif s.endswith('K'):
        p = 1
    else:
        p = 0
    suffix_len = 1 if p > 0 else 0
    base = s[0:(len(s) - suffix_len)]
    return int(base) * 1024**p


parser = argparse.ArgumentParser(
        description="Invoke Cahn-Hilliard CFD problem")
parser.add_argument("mesh_size", type=int,
        help="Size of the unit square mesh along one dimension")
parser.add_argument("ranks", type=int,
        help="Number of total ranks (recorded along with timing measurements)")
parser.add_argument("ranks_per_node", type=int,
        help="Number of total ranks per node " + \
		"(recorded along with timing measurements)")
parser.add_argument("--tasks", default='mesh,setup,solve',
        help="Comma-separated list of steps to perform (mesh,setup,solve)")
parser.add_argument("--solution-out",
        help="Output filename where to save solution (PVD)")
parser.add_argument("--elapsed-out",
        help="Output filename where to save measured times (CSV)")
parser.add_argument("--degree", type=int, default=1,
        help="Degree of the problem")
parser.add_argument("--steps", type=int, default=1,
        help="Number of timesteps to solve")
parser.add_argument("--preconditioner", default='fieldsplit',
        help="Preconditioner to use")
parser.add_argument("--ksp", default='gmres',
        help="Solver to use")
parser.add_argument("--inner-ksp", default='preonly',
        help="Inner solver to use")
parser.add_argument("--max-iterations", type=int, default=1,
        help="Inner solver to use")
parser.add_argument("--lmbda", type=float, default=1.0e-02,
        help="Surface parameter")
parser.add_argument("--dt", type=float, default=5.0e-06,
        help="Time step")
parser.add_argument("--theta", type=float, default=0.5,
        help="Time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson")
parser.add_argument("--compute-norms", action='store_true',
        help="Compute and print norms")
parser.add_argument("--mem-per-node", type=size_from_string,
        help="Limit the total memory usage of all ranks on each node (MB)")
parser.add_argument("--dedicated-node-for-rank0", action='store_true',
        help="A whole node was dedicated to rank 0 by MPI launcher " +
            "(affects memory limits)")
parser.add_argument("--monitor-mem", type=float,
        help="Print peak memory usage every N seconds.")
parser.add_argument("--verbose", action='store_true',
        help="Enable extra logging")
args = parser.parse_args()

def get_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

comm = MPI.COMM_WORLD

if comm.rank == 0:
    print("rank 0 node: ", platform.node(), "pid", os.getpid(),
            "ranks", args.ranks, "(", comm.size, ")", \
            "ranks_per_node", args.ranks_per_node, \
            "mesh", args.mesh_size)

if args.mem_per_node is not None:
    mem_res = resource.RLIMIT_AS
    mem_per_node = args.mem_per_node
    def_soft_lim, def_hard_lim = resource.getrlimit(mem_res)
    if args.dedicated_node_for_rank0 and comm.rank == 0:
        # Assumes a rankfile that dedicates a whole node to rank 0
        soft_lim = mem_per_node
    else:
        soft_lim = mem_per_node // args.ranks_per_node
    resource.setrlimit(mem_res, (soft_lim, def_hard_lim))
    if comm.rank == 0 or comm.rank == 1:
            soft_lim = mem_per_node // min(args.ranks_per_node, comm.size - 1)
    else:
        soft_lim = mem_per_node // min(args.ranks_per_node, comm.size)
    resource.setrlimit(mem_res, (soft_lim, def_hard_lim))
    if comm.rank == 0 or comm.rank == 1:
        print("rank", comm.rank, "node ", platform.node(),
                "Node Mem =", mem_per_node // 1024**2, "MB",
                "Per Rank:",
                "Soft Lim:", def_soft_lim // 1024**2, "->",
                soft_lim // 1024**2, "MB",
                "Hard Lim", def_hard_lim // 1024**2, "->",
                def_hard_lim // 1024**2, "MB")

if args.monitor_mem is not None and (comm.rank == 0 or comm.rank == 1):
    # Note: neither threading.Timer nor signal is perfect: apparently neither
    # can't preempt execution of C code, so the output may stop, but it appears
    # to be working just barely enough.
    import threading
    class MemMon:
        def __init__(self, interval):
            self.interval = interval
            self.active = True
            self.print_peak_mem()

        def print_peak_mem(self):
            res = resource.getrusage(resource.RUSAGE_SELF)
            peak_mem_mb = res.ru_maxrss / 1024
            print("rank", comm.rank, "node ", platform.node(),
                "mesh", args.mesh_size,
                "ranks", comm.size, "ranks_per_node", args.ranks_per_node,
                "peak mem", peak_mem_mb, "MB")
            self.mem_mon_timer = threading.Timer(self.interval, \
                    self.print_peak_mem)
            if self.active:
                self.mem_mon_timer.start()

        def stop(self):
            self.active = False
            self.mem_mon_timer.cancel()
            self.mem_mon_timer = None

    mem_mon = MemMon(args.monitor_mem)

peak_mem = OrderedDict()
peak_mem['init'] = get_mem_mb()
if comm.rank == 0 or comm.rank == 1:
    print("rank %u: init: peak mem %.0f MB" % (comm.rank, peak_mem['init']))

tasks = args.tasks.split(",")
if 'setup' in tasks:
    assert 'mesh' in tasks
if 'solve' in tasks:
    assert 'setup' in tasks

# We want to print progress as we go for purposes of debugging failed jobs,
# but we also want all output collected by rank at the end for easy viewing.
log_lines = []

peak_mem_pre_alloc = get_mem_mb()
print("rank", comm.rank, "node ", platform.node(),
    "mesh", args.mesh_size,
    "ranks", comm.size, "ranks_per_node", args.ranks_per_node,
    "mem prior to allocation", peak_mem_pre_alloc, "MB")

if 'mesh' in tasks:
    time_mesh_begin = time.time()
    try:
        mesh = CahnHilliardProblem.make_mesh(args.mesh_size)
    except MemoryError:
        peak_mem_at_failure = get_mem_mb()
        print("rank", comm.rank, "node ", platform.node(),
            "mesh", args.mesh_size,
            "ranks", comm.size, "ranks_per_node", args.ranks_per_node,
            "mem at failure", peak_mem_at_failure, "MB")
        raise
    time_mesh_end = time.time()
    peak_mem['mesh'] = get_mem_mb()
    if comm.rank == 0 or comm.rank == 1:
        breakdown = ";".join(["%s=%.2f s/MB" % (m, v) \
                for m, v in timings.items()])
        log_line = \
            "rank %u: step mesh took: %.2f s, peak mem %.0f MB: %s" % \
            (comm.rank, time_mesh_end - time_mesh_begin,
                peak_mem['mesh'], breakdown)
        log_lines.append(log_line)
        print(log_line)

if 'setup' in tasks:
    time_setup_begin = time.time()
    init_loop, mass_loops, hats_loops, assign_loops, u, u0, solver = \
            CahnHilliardProblem.do_setup(mesh, pc=args.preconditioner,
            degree=args.degree, dt=args.dt, theta=args.theta,
            lmbda=args.lmbda,
            ksp=args.ksp, inner_ksp=args.inner_ksp,
            maxit=args.max_iterations, verbose=args.verbose,
            out_lib_dir=os.path.join(os.getcwd(), 'ch_build'))
    time_setup_end = time.time()
    peak_mem['setup'] = get_mem_mb()
    if comm.rank == 0 or comm.rank == 1:
        log_line = \
                "rank %u: step setup took: %.2f s, peak mem %.0f MB" % \
                (comm.rank, time_setup_end - time_setup_begin,
                peak_mem['setup'])
        log_lines.append(log_line)
        print(log_line)

if 'solve' in tasks:
    # Output file
    if args.solution_out is not None:
        file = File(args.solution_out)
    else:
        file = None

    time_solve_begin = time.time()
    CahnHilliardProblem.do_solve(init_loop, mass_loops, hats_loops,
            assign_loops, u, u0, solver, args.steps,
            inner_ksp=args.inner_ksp, maxit=args.max_iterations,
            compute_norms=args.compute_norms, out_file=file)
    time_solve_end = time.time()
    peak_mem['solve'] = get_mem_mb()
    if comm.rank == 0 or comm.rank == 1:
        log_line = \
                "rank %u: step solve took: %.2f s, peak mem %.0f MB" % \
                (comm.rank, time_solve_end - time_solve_begin,
                peak_mem['solve'])
        log_lines.append(log_line)
        print(log_line)

if args.elapsed_out is not None:
    measurements = OrderedDict()
    measurements["mesh"] = args.mesh_size
    measurements["ranks"] = args.ranks
    measurements["ranks_per_node"] = args.ranks_per_node
    measurements["mesh_s"]  = time_mesh_end - time_mesh_begin \
            if 'mesh' in tasks else np.nan
    measurements['mesh_peakmem_mb'] = peak_mem.get('mesh', np.nan)
    measurements["setup_s"] = time_setup_end - time_setup_begin \
            if 'setup' in tasks else np.nan
    measurements['setup_peakmem_mb'] = peak_mem.get('setup', np.nan)
    measurements["solve_s"] = time_solve_end - time_solve_begin \
            if 'solve' in tasks else np.nan
    measurements['solve_peakmem_mb'] = peak_mem.get('solve', np.nan)
    measurements["total_s"] = measurements['mesh_s'] + \
            measurements['setup_s'] + measurements.get('solve_s', np.nan) \
            if 'mesh' in tasks and 'setup' in tasks and 'solve' in tasks \
            else np.nan

    # Prevent interleaving of printed lines among ranks: make ranks > 0 wait
    if comm.rank != 0:
        comm.barrier()

    if comm.rank == 0 or comm.rank == 1:
        for m, v in timings.items():
            measurements[m] = v
        for l in log_lines:
            print(l)
        for m, v in measurements.items():
            print("rank %u: %40s: %8.2f" % (comm.rank, m, v))

    if comm.rank == 0:
        comm.barrier()

    if comm.rank == 0:
        # note: if you open this earlier, the FD breaks somehow (???)
        fout = open(args.elapsed_out, "w")
        print("rank," + ",".join(measurements.keys()), file=fout)

        def save_measurements_row(rank, measurements):
            print(f"{rank}," + ",".join([str(v) for v in measurements.values()]), file=fout)

    if comm.rank != 0:
        comm.send(measurements, dest=0)
    else:
        save_measurements_row(comm.rank, measurements)
        for rank in range(1, comm.size):
            measurements_other = comm.recv(source=rank)
            save_measurements_row(rank, measurements_other)

    if comm.rank == 0:
        fout.close()

if mem_mon is not None:
    mem_mon.stop()
