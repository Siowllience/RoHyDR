
from run.run import ROHYDR_run

ROHYDR_run(model_name='rohydr',
           dataset_name='mosi',
           seeds=[5,6,7,8,9,10],
           mr=0.7)
