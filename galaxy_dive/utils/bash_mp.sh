#!/bin/bash

#SBATCH --job-name=ahf_m12m
#SBATCH --partition=skx-normal
## Stampede node has 16 processors & 32 GB
## Except largemem nodes, which have 32 processors & 1 TB
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/05779/tg850788/m12m_res57000/output/AHF/jobs/%j.out
#SBATCH --error=/scratch/05779/tg850788/m12m_res57000/output/AHF/jobs/%j.err
#SBATCH --mail-user=zhafen@u.northwestern.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST140023

########################################################################
# Input Arguments
########################################################################

# What snapshots to use
# snap_snum_start should not usually be lower than 1, because AHF handles
# snapshot 0 weirdly and there aren't usually halos there anyways
snap_num_start=1
snap_num_end=600
snap_step=1

# How many processors to use? (Remember to account for memory constraints)
n_procs=9

# Example arguments you might want to pass
data_dir=/some/place/thats/backed/up
save_dir=/some/place/that/is/also/backed/up/I/suppose

# Actual multiprocessing command
seq $snap_num_start $snap_step $snap_num_end | xargs -n 1 -P $n_procs sh -c 'python script_to_run.py $0 $1 $2' $data_dir $save_dir
# The above command is pretty complicated, so let me explain below, to the best of my abilities.
# Everything to the left of the pipe is setting up a sequence of numbers, with some special formatting.
# It will pass numbers from the sequence to xargs
# To the right we have xargs, which receives the things to the left and starts up multiprocessing on $n_procs cores.
# After the sh in to the right is setting up its own mini command window. It receives the piped number as the last command.
# In this case that means that the snapshot we want to run the multiprocessing for is passed as the $2 argument.
# If we had an additional argument besides $data_dir and $save_dir, then it would be passed as the $3 argument.
