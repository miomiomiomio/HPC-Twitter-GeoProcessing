#!/bin/bash
#SBATCH --partition=physical
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=1:00:00
#SBATCH --output=2n8c.txt

module load Python/3.5.2-goolf-2015a
time mpiexec python Project11.py

