"""Loads configuration yaml and runs an experiment."""
from argparse import ArgumentParser
import os
import glob
from datetime import datetime
import shutil
import yaml
from tqdm import tqdm
import torch
import numpy as np
from collections import defaultdict

import data
import model
import probe
import regimen
import reporter
import task
import loss
import glob
import sys
import random

from run_experiment import choose_dataset_class, choose_probe_class, choose_model_class



  
def evaluate_vectors(args, probe, dataset, model, results_dir, output_name, is_random):
  dataloader = dataset.get_dev_dataloader()
  

  relations_to_projections = defaultdict(list)
  for data_batch, label_batch, length_batch, observation_batch in dataloader:
    for label, length, (observation, _) in zip(label_batch, length_batch, observation_batch):
      # print(observation.embeddings)
      # print(" ".join(observation.sentence))
      # print(observation.sentence)
      # print(observation.head_indices)
      for idx, word in enumerate(observation.sentence):
        # print(idx)
        # print("Considering {}".format(word))
        # print("Projection is: {}".format(projection[idx]))
        if not is_random and observation.head_indices[idx] == '0' or len(observation.sentence) == 1:
          # print("Head word.")
          pass
        else:
          if is_random:
            head_index = random.choice([i for i in range(len(observation.sentence)) if i != idx])
          else:
            head_index = int(observation.head_indices[idx])
          
          # print("Head index is: {}".format(head_index))
          # print("Head word is: {}".format(observation.sentence[head_index-1]))
          # print("Relation is: {}".format(observation.governance_relations[idx]))
          proj_diff = observation.embeddings[idx] - observation.embeddings[head_index-1]
          # print(proj_diff.shape)
          relations_to_projections[observation.governance_relations[idx]].append(proj_diff)
  relations_to_diffs = {}
  all_relations = []
  y_list = []
  for relation in relations_to_projections:
    # print(relation, len(relations_to_projections[relation]))
    if len(relations_to_projections[relation]) > 100:
      print(relation)
      diffs = torch.stack(relations_to_projections[relation])
      # print(diffs.shape)
      # compute the SVD
      u, s, v = diffs.svd()
      # print(s)
      average_diff = torch.mean(diffs, 0)
      relations_to_diffs[relation] = average_diff
      all_relations += relations_to_projections[relation]
      y_list += [relation] * len(relations_to_projections[relation])
  allDiff = torch.stack(all_relations)
  # print(y_list)
  if len(sys.argv) > 2:
    np.save('/sailhome/ethanchi/structural-probes/relationOutputs/{}-bert.npy'.format(output_name), allDiff.numpy())
    np.save('/sailhome/ethanchi/structural-probes/relationOutputs/{}-bertY.npy'.format(output_name), np.array(y_list))

  # allDiff = torch.mean(allDiff, 0)
  cos = torch.nn.CosineSimilarity(dim=0, eps=1e-10)
  for relation in relations_to_diffs:
    print(relation, torch.norm(relations_to_diffs[relation]))
    # for relation2 in relations_to_diffs:
        # print(relation, relation2, cos(relations_to_diffs[relation], relations_to_diffs[relation2]))
  # print("AllDiff", torch.norm(allDiff))
        # print("Projection is: {}".format(projection[int(observation.head_indices[idx])-1]))


  #train_dataloader = dataset.get_train_dataloader(shuffle=False)
  #train_predictions = regimen.predict(probe, model, train_dataloader)
  #reporter(train_predictions, train_dataloader, 'train')

  # Uncomment to run on the test set
  #test_dataloader = dataset.get_test_dataloader()
  #test_predictions = regimen.predict(probe, model, test_dataloader)
  #reporter(test_predictions, test_dataloader, 'test')

def execute_experiment(args, results_dir, output_name, is_random):
  """
  Execute an experiment as determined by the configuration
  in args.

  Args:
    train_probe: Boolean whether to train the probe
    report_results: Boolean whether to report results
  """
  dataset_class = choose_dataset_class(args)
  # task_class, reporter_class, loss_class = choose_task_classes(args)
  probe_class = choose_probe_class(args)
  model_class = choose_model_class(args)
#  regimen_class = regimen.ProbeRegimen

  expt_dataset = dataset_class(args, task.DummyTask)
  # expt_reporter = reporter_class(args)
  expt_probe = probe_class(args)
  expt_model = model_class(args)
#  expt_regimen = regimen_class(args)
#  expt_loss = loss_class(args)

  evaluate_vectors(args, expt_probe, expt_dataset, expt_model, results_dir, output_name, is_random)


if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('dir')
  argp.add_argument('output_name')
  argp.add_argument('-r', dest='random', action='store_true')
  argp.add_argument('--seed', default=0, type=int,
      help='sets all random seeds for (within-machine) reproducibility')
  cli_args = argp.parse_args()
  if cli_args.seed:
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  # yaml_args = yaml.load(open(#")
  os.chdir(cli_args.dir)
  yaml_args = yaml.load(open(glob.glob("*.yaml")[0]))
  # setup_new_experiment_dir(cli_args, yaml_args, cli_args.results_dir)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  yaml_args['device'] = device
  if cli_args.random: print("Using random.")
  execute_experiment(yaml_args, cli_args.dir, cli_args.output_name, cli_args.random)
