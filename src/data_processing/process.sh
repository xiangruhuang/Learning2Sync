#!/bin/bash
dataset=$1
for i in `cat ${dataset}.list`; do
    python process_${dataset}.py --shapeid $i;
done
