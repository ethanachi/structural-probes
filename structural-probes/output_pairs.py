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

from run_experiment import choose_dataset_class, choose_probe_class, choose_model_class


def load_projected_representations(probe, model, dataset):
  """
  Loads projected representations under `probe` from `dataset`.
  """
  projections_by_batch = []
  for batch in tqdm(dataset, desc='[predicting]'):
    observation_batch, label_batch, length_batch, _ = batch
    word_representations = model(observation_batch)
    transformed_representations = torch.matmul(word_representations, probe.proj)
    projections_by_batch.append(transformed_representations.detach().cpu().numpy())
  return projections_by_batch
  
def evaluate_vectors(args, probe, dataset, model, results_dir, output_name):
  probe_params_path = os.path.join(results_dir, args['probe']['params_path'])
  probe.load_state_dict(torch.load(probe_params_path))
  probe.eval()
  print(probe.proj)
  
  dataloader = dataset.get_dev_dataloader()
  
  projections = load_projected_representations(probe, model, dataloader)

  all_pairs = []
  all_labels = []
  all_sentences = []
  all_idxs = []
  all_words = []

  relations_to_pairs = defaultdict(list)
  relations_to_labels = defaultdict(list)
  relations_to_sentences = defaultdict(list)
  relations_to_idxs = defaultdict(list)
  relations_to_words = defaultdict(list)

  for projection_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(projections, dataloader):
    for projection, label, length, (observation, _) in zip(projection_batch, label_batch, length_batch, observation_batch):
      for idx, word in enumerate(observation.sentence):
        if observation.head_indices[idx] == '0':
          pass # head word 
        else:
          head_index = int(observation.head_indices[idx])
          relation = observation.governance_relations[idx]

          relations_to_pairs[relation].append((projection[idx], projection[head_index-1]))
          relations_to_labels[relation].append((observation.governance_relations[idx]))
          relations_to_sentences[relation].append(" ".join(observation.sentence))
          relations_to_idxs[relation].append(idx)
          relations_to_words[relation].append(word)

  for relation in relations_to_pairs:
    all_pairs += relations_to_pairs[relation]
    all_labels += relations_to_labels[relation]
    all_sentences += relations_to_sentences[relation]
    all_idxs += relations_to_idxs[relation]
    all_words += relations_to_words[relation]

  all_pairs = np.array(all_pairs)
  print("Pairs shape:", all_pairs.shape)
  all_labels = np.array(all_labels)
  print("Labels shape:", all_labels.shape)
  sentences_idxs_words = np.array([all_sentences, all_idxs, all_words]).T
  print("Data shape:", sentences_idxs_words.shape)
  if len(sys.argv) > 2:
    np.save('/sailhome/ethanchi/structural-probes/relationOutputs/{}-pairs.npy'.format(output_name), all_pairs)
    np.save('/sailhome/ethanchi/structural-probes/relationOutputs/{}-pairs-Y.npy'.format(output_name), all_labels)
    np.save('/sailhome/ethanchi/structural-probes/relationOutputs/{}-pairs-data.npy'.format(output_name), sentences_idxs_words)
  # allDiff = torch.mean(allDiff, 0)
  cos = torch.nn.CosineSimilarity(dim=0, eps=1e-10)
  # for relation in relations_to_diffs:
    # print(relation, torch.norm(relations_to_diffs[relation]))
    # for relation2 in relations_to_diffs:
        # print(relation, relation2, cos(relations_to_diffs[relation], relations_to_diffs[relation2]))
  # print("AllDiff", torch.norm(allDiff))
        # print("Projection is: {}".format(projection[int(observation.head_indices[idx])-1]))

def execute_experiment(args, results_dir, output_name):
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

  evaluate_vectors(args, expt_probe, expt_dataset, expt_model, results_dir, output_name)


if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('dir')
  argp.add_argument('output_name')
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
  execute_experiment(yaml_args, cli_args.dir, cli_args.output_name)
