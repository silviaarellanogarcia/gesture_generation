#!/bin/bash

#SBATCH --chdir=/home/silviaag/gesticulator/gesticulator
#SBATCH --job-name=training.sh
#SBATCH --output=output.sh.o%j
#SBATCH --error=output.sh.e%j

python train.py