import numpy as np
import scipy.io as sio

for i in range(1, 4):
    results = sio.loadmat('../results/mvkdr/simu/simu%s.mat' % i)
    print 'Simu%s mvkdr  ACC: %.1f NMI: %.1f' % (i, results['acc'][0][0]*100, results['nmi'][0][0]*100)   

    print

uci = ['hepatitis', 'iris', 'wine', 'glass', 'ionosphere', 'wdbc']

for data in uci:
    results = sio.loadmat('../results/mvkdr/uci/%s.mat' % data)
    print '%s mvkdr  ACC: %.1f NMI: %.1f' % (data, results['acc'][0][0]*100, results['nmi'][0][0]*100)    

    print

webkb = ['Cornell', 'Texas', 'Washington', 'Wisconsin', 'all']

for data in webkb:
    results = sio.loadmat('../results/mvkdr/webkb/%s.mat' % data)
    print '%s mvkdr  ACC: %.1f NMI: %.1f' % (data, results['acc'][0][0]*100, results['nmi'][0][0]*100)

    print    
