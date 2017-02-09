#!/bin/bash

for i in 1 2 3
do
    for j in 0 1 2 3 4 5 6 7 8 9
    do
        for th1 in 0.01 0.1 1 10 100
        do
            for th2 in 0.01 0.1 1 10 100
            do
                python mvkdr_simu.py --simu $i --seed $j --theta1 $th1 --theta2 $th2 --output ../results/mvkdr/simu/
            done
        done
    done
    matlab -nodisplay -nosplash -nodesktop -r "result_mvkdr('../results/mvkdr/simu/simu$i');"
done