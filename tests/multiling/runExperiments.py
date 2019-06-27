import os

print('nlprun -a sp -n run-sp-experiments ', end='')

for experiment in sorted(os.listdir('experiments')):
  print("'python ~/structural-probes/structural-probes/run_experiment.py ~/structural-probes/tests/multiling/experiments/{}'".format(experiment))