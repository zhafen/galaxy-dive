#!/bin/bash

# Generate docs using the docstrings
sphinx-apidoc -o . ../galaxy_dive -e -f

# Make the files
make html
