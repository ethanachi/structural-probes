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
import h5py
import getpass

from sklearn.manifold import TSNE

import data
import model
import probe
import regimen
import reporter
import task
import loss
import glob
import sys
import os

from run_experiment import choose_dataset_class, choose_probe_class, choose_model_class

def save_vector(path, name, arr):
  np.save(os.path.join(path, name + '.npy'), arr)

def write_data(args, probe, dataset, model, results_dir, output_path):
  probe_params_path = os.path.join(results_dir, args['probe']['params_path'])
  probe.load_state_dict(torch.load(probe_params_path))
  probe.eval()


  for dataloader, split_name in zip((dataset.get_train_dataloader(), dataset.get_dev_dataloader()), ("train", "dev")):

    to_output = ["projections", "sentences", "idxs", "words", "relations", "pos", "pairs", "diffs", "morphs", "representations", "is_head"]
    outputs = defaultdict(list)

    i = 0
    for data_batch, label_batch, length_batch, observation_batch in dataloader:
      for label, length, (observation, _), representation in zip(label_batch, length_batch, observation_batch, data_batch):
        representation = model(representation[:length])
        proj_matrix = probe.proj if hasattr(probe, 'proj') else probe.linear1.weight.data.transpose(0, 1)
        projection = torch.matmul(representation, proj_matrix).detach().cpu().numpy()
        head_indices = [int(x) - 1 for x in observation.head_indices]
        projection_heads = projection[head_indices]
        to_add = {
          "projections": projection,
          "sentences": [" ".join(observation.sentence)] * int(length),
          "idxs": range(representation.shape[0]),
          "words": observation.sentence,
          "relations": observation.governance_relations,
          "pos": (observation.upos_sentence if hasattr(observation, 'upos_sentence') else observation.pos),
          "pairs": np.stack((projection, projection_heads)),
          "diffs": np.array(projection) - np.array(projection_heads),
          "is_head": [(x == '0') for x in observation.head_indices],
          "labels": label[:length].detach().cpu().numpy()
        }
        if split_name == "": to_add['representations'] = representation.detach().cpu().numpy(),
        if hasattr(observation, 'morph'): to_add['morphs'] = observation.morph
        if hasattr(observation, 'pred'): to_add['pred'] = observation.pred
        if hasattr(probe, 'linear1'): to_add['logits'] = probe(representation.unsqueeze(0)).detach().cpu().numpy().squeeze(axis=0)
        for target in to_add:
          outputs[target] += list(to_add[target])
        i += 1

    for output in outputs:
      if output in ('representations', 'projections', 'logits'):
        hf = h5py.File(os.path.join(output_path, split_name + '-' + output + '.hdf5'), 'w')
        hf.create_dataset(output, data=np.array(outputs[output]), compression="gzip")
        hf.close()
      else:
        outputs[output] = np.array(outputs[output])
        save_vector(output_path, split_name + '-' + output, outputs[output])

  return outputs

def perform_tsne(outputs, to_write, output_path, num_to_write=10000):
  tsne = TSNE(n_components=2, random_state=229, verbose=10)
  print("Fitting TSNE.")

  if USE_MULTILINGUAL:
    # we sample uniformly per-language
    langs = lang_mapping.keys()
    langs_to_indices = {}
    num_needed = num_to_write // len(langs)
    for lang in langs:
      selector = np.vectorize(lambda x: x.startswith(lang + '-'))
      indices_to_choose = np.where(selector(all_labels))[0]
      indices_needed = indices_to_choose[np.random.choice(indices_to_choose.shape[0], num_needed, replace=False)]
      langs_to_indices[lang] = indices_needed
    indices = np.concatenate([langs_to_indices[lang] for lang in langs])
  else:
    indices = np.random.choice(projections.shape[0], num_needed, replace=False)

  cut_outputs = dict(((output_name, outputs[output_name][indices]) for output_name in to_write))
  cut_diffs = outputs['diffs'][indices]

  reduced = tsne.fit_transform(cut_diffs)
  print("Fitted.")

  tsv_out = np.stack((reduced, *cut_vectors.values()), axis=1)
  for cut_output in cut_outputs:
    save_vector(output_path, cut_output + '-cut', output_vector)
  save_vector(output_path, 'reduced', reduced)

  header = "x0\tx1\t" + "\t".join(cut_output)
  np.savetxt(os.path.join(output_path, 'data.tsv'), tsv_out, fmt="%s", header=header, comments="", delimiter="\t")



def execute_experiment(args, results_dir, output_name, use_tsne):
  """
  Execute an experiment as determined by the configuration
  in args.

  Args:
    train_probe: Boolean whether to train the probe
    report_results: Boolean whether to report results
  """
  dataset_class = choose_dataset_class(args)
  probe_class = choose_probe_class(args)
  model_class = choose_model_class(args)

  expt_dataset = dataset_class(args, task.SemanticRolesTask)
  expt_probe = probe_class(args)
  expt_model = model_class(args)

  outputs = write_data(args, expt_probe, expt_dataset, expt_model, results_dir, output_path)
  if use_tsne: perform_tsne(outputs["projections"])
  return outputs

if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('dir')
  argp.add_argument('output_name')
  argp.add_argument('--seed', default=0, type=int,
      help='sets all random seeds for (within-machine) reproducibility')
  argp.add_argument('--tsne', dest="tsne", action="store_true")

  cli_args = argp.parse_args()
  if cli_args.seed:
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  output_path = '/u/scr/{}/relationOutputs/{}'.format(getpass.getuser(), cli_args.output_name)

  try:
    os.mkdir(output_path)
  except FileExistsError:
    # print("Error: output directory {} already exists.".format(output_path))
    pass

  os.chdir(cli_args.dir)
  yaml_args = yaml.load(open(glob.glob("*.yaml")[0]))
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  yaml_args['device'] = device
  execute_experiment(yaml_args, cli_args.dir, output_path, cli_args.tsne)
