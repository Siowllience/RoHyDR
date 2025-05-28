
from run.run import ROHYDR_run

ROHYDR_run(model_name='rohydr',
           dataset_name='mosi',
           seeds=[1,2,3],
           mr=0.1)
