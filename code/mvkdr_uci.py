import math
import logging
import argparse
import numpy as np
import scipy.io as sio

from mvkdr import mvkdr
from utility import mediandist
from uci import *

parser = argparse.ArgumentParser(description="Experiments MvKDR Simulation")
parser.add_argument("--data", required=True, 
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

output = '%s/%s_seed%s_th1%s_th2%s' % (args.output, args.data, args.seed, args.theta1, args.theta2)
seed = int(args.seed)
theta1 = float(args.theta1)
theta2 = float(args.theta2)

if args.data == 'iris':
    X1, X2, y = iris('../data/uci/iris.data')
elif args.data == 'glass':
    X1, X2, y = glass('../data/uci/glass.data')
elif args.data == 'ionosphere':
    X1, X2, y = ionosphere('../data/uci/ionosphere.data')
elif args.data == 'hepatitis':
    X1, X2, y = hepatitis('../data/uci/hepatitis.data')
elif args.data == 'wine':
    X1, X2, y = wine('../data/uci/wine.data')    
elif args.data == 'wdbc':
    X1, X2, y = wdbc('../data/uci/wdbc.data')    
else:
    print 'Wrong UCI data!'
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