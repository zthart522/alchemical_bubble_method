#!/bin/bash

module purge
source /home1/11346/zthart/codes/chimes_calculator/modfiles/UT-TACC.mod

export FI_PROVIDER=tcp
export OMP_NUM_THREADS=4

{{ cmd }}
