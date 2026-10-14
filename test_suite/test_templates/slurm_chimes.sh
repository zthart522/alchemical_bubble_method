#!/bin/bash
#SBATCH -N {{ nodes }}
#SBATCH -p {{ queue }}
#SBATCH -A {{ account }}
#SBATCH -t {{ walltime }}

module purge
source /home1/11346/zthart/codes/chimes_calculator/modfiles/UT-TACC.mod

ibrun {{ cmd }}