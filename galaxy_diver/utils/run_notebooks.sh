#!/bin/bash

for nb in *.ipynb;
do
    # This doesn't work with Python2.7 (assumed default, so switch to Python 3)
    # module unload python
    # module load python3

    # Convert NBs
    jupyter nbconvert --to python $nb
    nb_script=${nb/.ipynb/.py}
    echo Converted $nb to $nb_script ...

    # Switch back to Python 2
    # module load python/2.7.13
    # module unload python3

    # And now run the NBs
    echo Running $nb_script ...
    python $nb_script

    # Clean up by removing scripts.
    # rm $nb_script
    
done
