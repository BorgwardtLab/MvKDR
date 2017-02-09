import math
import logging
import argparse
import numpy as np
import scipy.io as sio

from mvkdr import mvkdr
from utility import mediandist
from simulation import simulation

parser = argparse.ArgumentParser(description="Experiments MvKDR Simulation")
parser.add_argument("--simu", required=True, 
                    help="simulation data set")
parser.add_argument("--seed", required=True, 
                    help="random seed")
parser.add_argument("--output", required=True, 
                    help="output folder saving results")
parser.add_argument("--theta1", required=True, 
                    help="regularization parameter")
parser.add_argument("--theta2", required=True, 
                    help="regularization parameter")
args = parser.parse_args()

output = '%s/simu%s_seed%s_th1%s_th2%s' % (args.output, args.simu, args.seed, args.theta1, args.theta2)
seed = int(args.seed)
theta1 = float(args.theta1)
theta2 = float(args.theta2)

if args.simu == '1':
    X1, X2, y = simulation(seed, 0.5, 0.5)
elif args.simu == '2':
    X1, X2, y = simulation(seed, 0.5, 0.25)
elif args.simu == '3':
    X1, X2, y = simulation(seed, 0.75, 0.25)
else:
    print 'Wrong simulation data!'
    exit(0)

k = np.unique(y).shape[0]    

sigma1 = math.sqrt(mediandist(X1))
sigma2 = math.sqrt(mediandist(X2))

y_pred, km_obj = mvkdr(X1, X2, k, sigma1, sigma2, theta1, theta2, seed)

result = {}
result['y'] = y
result['y_pred'] = y_pred
result['km_obj'] = km_obj

sio.savemat('%s.mat' % output, result)
np.savez(output, y=y, y_pred=y_pred, km_obj=km_obj)