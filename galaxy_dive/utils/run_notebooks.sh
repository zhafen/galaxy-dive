#!/bin/bash

#SBATCH --job-name=nb_runner
#SBATCH --partition=skx-normal
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=0:45:00
#SBATCH --output=/scratch/03057/zhafen/pathfinder_data/job_scripts/jobs/%j.out
#SBATCH --error=/scratch/03057/zhafen/pathfinder_data/job_scripts/jobs/%j.err
#SBATCH --mail-user=zhafen@u.northwestern.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST140023

nbs_to_run=(*.ipynb)
# nbs_to_run=(mass_budget.ipynb)

for nb in ${nbs_to_run[@]};
do
    # This doesn't work with Python2.7
    # module unload python
    # module load python3

    # Convert NBs
    jupyter nbconvert --to python $nb
    nb_script=${nb/.ipynb/.py}
    echo Converted $nb to $nb_script ...

    # Switch back to Python 2
    # module load python/2.7.13
    # module unload python3
    # module load phdf5

    # And now run the NBs
    echo Running $nb_script ...
    python $nb_script "$@"

    # Clean up by removing scripts.
    rm $nb_script
    
done
