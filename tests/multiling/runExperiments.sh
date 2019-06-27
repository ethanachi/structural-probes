#!/bin/bash

fname="sl/$(date +"%m-%d-%Y-%H-%M.txt")"
echo $fname
for f in experiments/*.sh;
do
  id=$(sbatch $f | cut -d ' ' -f 4)
  echo "$fname\t$id" >> $fname
done
