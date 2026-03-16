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
echo "Log: Start cloning repo"
git clone https://github.com/NeuralEnsemble/elephant.git $TMPDIR/ele-nbeval
cd $TMPDIR/ele-nbeval

# Setup env
echo "Log: Start env setup"
python -m venv .venv
source .venv/bin/activate
echo "Log: Start pip install"
printenv PATH
pip install -e .[tests,docs,extras,tutorials]
pip install --upgrade matplotlib==3.10.8
pip install nbval

# List packages
pip list


echo "Print matplot version"
python -c "import matplotlib; print(matplotlib.__version__)"
