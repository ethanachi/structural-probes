"""Loads configuration yaml and runs an experiment."""
from argparse import ArgumentParser
import os
from datetime import datetime
import shutil
import yaml
from tqdm import tqdm
import torch
import numpy as np

import data
import model
import probe
import regimen
import reporter
import task
import loss

from run_experiment import choose_dataset_class, choose_probe_class, choose_model_class


def load_projected_representations(probe, model, dataset):
  """
  Loads projected representations under `probe` from `dataset`.
  """
  representations_by_batch = []
  for batch in tqdm(dataset, desc='[predicting]'):
    observation_batch, label_batch, length_batch, _ = batch
    word_representations = model(observation_batch)
    transformed_representations = torch.matmul(batch, probe.proj)
    predictions_by_batch.append(predictions.detach().cpu().numpy())
  return representations_by_batch
  
def evaluate_vectors(args, probe, dataset, model):
  probe_params_path = os.path.join(args['reporting']['root'], args['probe']['params_path'])
  probe.load_state_dict(torch.load(probe_params_path))
  probe.eval()
  
  dataloader = dataset.get_dev_dataloader()
  
  projections = load_projected_representations(probe, model, dev_dataloader)
  for projection_batch, (data_batch, label_batch, length_batch, observation_batch) in tqdm(zip(projections, dataloader)):
    for projection, label, length, (observation, _) in zip(projection_batch, label_batch, length_batch, observation_batch):
      print(observation.sentence)


  #train_dataloader = dataset.get_train_dataloader(shuffle=False)
  #train_predictions = regimen.predict(probe, model, train_dataloader)
  #reporter(train_predictions, train_dataloader, 'train')

  # Uncomment to run on the test set
  #test_dataloader = dataset.get_test_dataloader()
  #test_predictions = regimen.predict(probe, model, test_dataloader)
  #reporter(test_predictions, test_dataloader, 'test')

def execute_experiment(args):
  """
  Execute an experiment as determined by the configuration
  in args.

  Args:
    train_probe: Boolean whether to train the probe
    report_results: Boolean whether to report results
  """
  dataset_class = choose_dataset_class(args)
  task_class, reporter_class, loss_class = choose_task_classes(args)
  probe_class = choose_probe_class(args)
#  model_class = choose_model_class(args)
#  regimen_class = regimen.ProbeRegimen

  task = task_class()
  expt_dataset = dataset_class(args, task)
  expt_reporter = reporter_class(args)
  expt_probe = probe_class(args)
#  expt_model = model_class(args)
#  expt_regimen = regimen_class(args)
#  expt_loss = loss_class(args)

  evaluate_vectors(args, expt_probe, expt_dataset, expt_model)




if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('dir')
  argp.add_argument('--seed', default=0, type=int,
      help='sets all random seeds for (within-machine) reproducibility')
  cli_args = argp.parse_args()
  if cli_args.seed:
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  yaml_args= yaml.load(open(cli_args.experiment_config))
  setup_new_experiment_dir(cli_args, yaml_args, cli_args.results_dir)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  yaml_args['device'] = device
  execute_experiment(yaml_args, train_probe=cli_args.train_probe, report_results=cli_args.report_results)
