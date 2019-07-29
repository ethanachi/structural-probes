#!/bin/bash


while read test
do
  # echo $test
  title=`echo $test | cut -d ' ' -f1`
  job_id=`echo $test | cut -d ' ' -f2`
   outputName="sl/slurm-${job_id}.out"
  # echo "Output name is" $outputName
  folder=`grep "Constructing new results" $outputName | grep -o 'experimentResults.*'`
  # echo $folder
  if [ -f $folder/dev.uuas ]; then 
    printf $title
    printf "\t"
    paste <(cat $folder/dev.uuas) <(cat $folder/dev.spearmanr-5_50-mean)
  fi
done < <(cat $1 | sort -t'-' -k1,1 -k3,3n -k4,4n)
