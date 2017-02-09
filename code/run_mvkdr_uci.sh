#!/bin/bash

for data in iris wine wdbc glass hepatitis ionosphere
do
    for j in 0 1 2 3 4 5 6 7 8 9
    do
        for th1 in 0.01 0.1 1 10 100
        do
        	for th2 in 0.01 0.1 1 10 100
        	do
	            python mvkdr_uci.py --data $data --seed $j --theta1 $th1 --theta2 $th2 --output ../results/mvkdr/uci/
	        done
        done
    done
    matlab -nodisplay -nosplash -nodesktop -r "result_mvkdr('../results/mvkdr/uci/$data');"
done