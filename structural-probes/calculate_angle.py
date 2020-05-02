# Takes in a text file formatted as such:
# ar  /u/scr/ethanchi/probingOutputs/blahblahblah
# fr  /u/scr/ethanchi/probingOutputs/blahblahblah2
# etc.
# and outputs a cross table

from argparse import ArgumentParser
import os
import yaml
import glob
from run_experiment import choose_probe_class
import torch
from scipy.linalg import subspace_angles
import itertools
from collections import defaultdict

argp = ArgumentParser()
argp.add_argument('source_files')
argp.add_argument('--seed', default=0, type=int,
    help='sets all random seeds for (within-machine) reproducibility')

cli_args = argp.parse_args()

with open(cli_args.source_files, 'r') as f:
  source_files = [x.strip().split('\t') for x in f]

proj_matrices = []
langs = []
for lang, source_file in source_files:
    os.chdir(source_file)
    args = yaml.load(open(glob.glob("*.yaml")[0]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args['device'] = device
    params_path = args['probe']['params_path']
    probe_class = choose_probe_class(args)
    probe = probe_class(args)
    probe.load_state_dict(torch.load(params_path))
    probe.eval()
    print("Evaluated", lang)
    proj_matrices.append(probe.proj.detach().cpu().numpy())
    langs.append(lang)

data = list(zip(langs, proj_matrices))

grid = defaultdict(dict)

for (lang1, u), (lang2, v) in itertools.product(data, data):
  grid[lang1][lang2] = subspace_angles(u, v).mean()

print('\t', end='')
for lang in langs:
  print(lang, end='\t')
print()

for lang1 in langs:
  print(lang1, end='\t')
  for lang2 in langs:
    print(f"{grid[lang1][lang2]:.3f}", end='\t')
  print()
