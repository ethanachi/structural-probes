#!/bin/bash

if [ $# -eq 2 ] && (($2 == "--chinese"))
then
  use_chinese="--use_chinese"
else
  use_chinese=""
fi
for f in $1/*.conllu
do 
  python3 ~/structural-probes/scripts/convert_conll_to_raw.py $f $use_chinese > "${f/.conllu/.txt}"
done
