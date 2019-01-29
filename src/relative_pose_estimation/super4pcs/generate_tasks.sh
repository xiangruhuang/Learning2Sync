#!/bin/bash

mkdir -p scannet
for sid in `cat scannet.list`; do 
    i=`echo ../../datasets/scannet/${sid}`; 
    echo $i; 
    for j in `seq 0 49`; do 
        inc=$(( j + 1)); 
        for k in `seq ${inc} 49`; do 
            name=${sid}_${j}_${k}; 
            name=../../datasets/scannet/pairwise/super4pcs/${name}.txt;
            echo $name
            if [[ -e ${name} ]]; then 
                continue; 
            fi;
            echo ./Super4PCS -i ${i}_${j}.obj ${i}_${k}.obj -o 0.7 -d 0.01 -t 1000 -n 200 -m ${name} 
            echo ./Super4PCS -i ${i}_${j}.obj ${i}_${k}.obj -o 0.7 -d 0.01 -t 1000 -n 200 -m ${name} >> scannet/tasks;
        done 
    done; 
done
