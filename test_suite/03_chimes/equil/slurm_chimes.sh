#!/bin/bash
#SBATCH -N 1
#SBATCH -p skx-dev
#SBATCH -A TG-CHM250015
#SBATCH -t 02:00:00

module purge
source /home1/11346/zthart/codes/chimes_calculator/modfiles/UT-TACC.mod

ibrun /home1/11346/zthart/codes/chimes_calculator/etc/lmp/exe/lmp_mpi_chimes  -in lammps.in | tee out.lammps