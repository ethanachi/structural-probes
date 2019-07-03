#!/bin/bash

fname="sl/$(date +"%m-%d-%Y-%H-%M.txt")"
echo $fname
for f in `find $1`;
do
  id=$(sbatch $f | cut -d ' ' -f 4)
  echo $f
  echo $f$'\t'$id >> $fname
done
