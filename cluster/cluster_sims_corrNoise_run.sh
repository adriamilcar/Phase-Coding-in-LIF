#!/bin/bash

# Choose a partition to run job in
#SBATCH --partition=snn

# Send output to simulations.log in current directory
#SBATCH --output simulations.log

# Defines number of jobs to execute. X% means maximum X jobs at the same time
#SBATCH --array 0-39999%80

# Defines maximum amount of memory to be used
#SBATCH --mem 4096

# Send email when job ends
#SBATCH --mail-type=END
#SBATCH --mail-user=<afernandez@science.ru.nl>

python3 -c "import cluster_sims_corrNoise; cluster_sims_corrNoise.grid_freqNoise_exp($SLURM_ARRAY_TASK_ID)"