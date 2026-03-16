#!/bin/bash

#SBATCH --job-name elephant-eval-nbs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:05:00
#SBATCH --partition=hamsteinZen3
#SBATCH --output=ele-eval-nbs-%j.out
#
# Load the MPI library
module load stable ias6

# Get latest elephant
git clone -b notebook-eval-nbval --single-branch https://github.com/INM-6/elephant.git $TMPDIR/ele-nbeval
cd $TMPDIR/ele-nbeval

# Setup env
python -m venv .venv
source .venv/bin/activate
printenv PATH
pip install -e .[tests,docs,extras,tutorials]
pip install nbval

echo "Print matplot version"
python -c "import matplotlib; print(matplotlib.__version__)"
