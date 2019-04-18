#!/bin/bash
#SBATCH --partition=physical
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --output=1n1c.txt

module load Python/3.5.2-goolf-2015a

time mpiexec python3 Project11.py

