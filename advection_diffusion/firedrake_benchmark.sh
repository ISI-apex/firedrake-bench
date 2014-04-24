#!/bin/bash
for np in 1 2 4
do
  echo Benchmarking Firedrake on $np processes
  echo =====================================
  echo
  mpirun -np $np python firedrake_advection_diffusion.py
  echo
done
