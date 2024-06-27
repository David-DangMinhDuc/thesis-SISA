import pandas as pd

import argparse

# Compute stats based on the number of data points that need to be retrained of the shards

parser = argparse.ArgumentParser()
parser.add_argument('--container', help="Name of the container")
args = parser.parse_args()

rp = pd.read_csv('containers/{}/numOfRetrainPoints.tmp'.format(args.container), names=['nb_retrained_points'])
print(rp['nb_retrained_points'].sum())
