#!/bin/bash

for f in $1/*.conllu; do python3 ~/structural-probes/scripts/convert_conll_to_raw.py "${f}" > "${f/.conllu/.txt}"; done
