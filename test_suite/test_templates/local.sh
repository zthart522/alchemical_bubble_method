#!/bin/bash

module purge
module load gcc/15.1.0
module load openmpi/5.0.8
module load hdf5/1.10.11

export FI_PROVIDER=tcp
export OMP_NUM_THREADS=4

{{ cmd }}
