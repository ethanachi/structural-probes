#!/bin/bash

for type in base-uncased large-uncased
do
    for f in $1/*.txt
    do
        echo "Processing $f $type"
        #echo "Path is: ${f/.txt/-$type.hdf5}"
        python3 ~/structural-probes/scripts/convert_raw_to_bert.py "$f" "${f/.txt/-$type.hdf5}" "$type" 
    done
done

