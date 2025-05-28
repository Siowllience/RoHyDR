"""
Training script for IMDER
dataset_name: Selecting dataset (mosi or mosei)
seeds: This is a list containing running seeds you input
mr: missing rate ranging from 0.1 to 0.7
"""
from run.run import ROHYDR_run

ROHYDR_run(model_name='rohydr',
           dataset_name='mosi',
           seeds=[1,2,3],
           mr=0.1)
