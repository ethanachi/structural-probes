#!/bin/bash

for f in experiments/*.sh;
do
  sbatch f
done
