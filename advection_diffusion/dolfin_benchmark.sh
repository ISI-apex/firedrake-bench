#!/bin/bash
for np in 1 2 4
do
  echo Benchmarking DOLFIN on $np processes
  echo ==================================
  echo
  mpirun -np $np python dolfin_advection_diffusion.py
  echo
done
