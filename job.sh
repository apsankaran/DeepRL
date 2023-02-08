#!/bin/bash

set -e

# 1. setup anaconda environment

# replace env-name on the right hand side of this line with the name of your conda environment

ENVNAME=research1

# if you need the environment directory to be named something other than the environment name, change this line

ENVDIR=$ENVNAME

# these lines handle setting up the environment; you shouldn't have to modify them

export PATH

mkdir $ENVDIR

tar -xzf $ENVNAME.tar.gz -C $ENVDIR

. $ENVDIR/bin/activate

# make sure the script will use your Python installation,

# and the working directory as its home location

export PATH=$PWD/python/bin:$PATH

export PYTHONPATH=$PWD/$ENVNAME

export HOME=$PWD

pip install mujoco gym

pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests

# run your script

python3 StableBaselines3.py
