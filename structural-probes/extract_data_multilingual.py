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

lang_mapping = {
  "ar": range(1 - 1, 909),
  "de": range(910 - 1, 1708),
  "en": range(1709 - 1, 3710),
  "es": range(3711 - 1, 5364),
  "fa": range(5365 - 1, 5963),
  "fi": range(5964 - 1, 7327),
  "fr": range(7328 - 1, 8803),
  "id": range(8804 - 1, 9362),
  "zh": range(9363 - 1, 9862)
}



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
  
def evaluate_vectors(args, probe, dataset, model, results_dir, output_name, use_tsne):
  probe_params_path = os.path.join(results_dir, args['probe']['params_path'])
  probe.load_state_dict(torch.load(probe_params_path))
  probe.eval()
  print(probe.proj)
  
  dataloader = dataset.get_dev_dataloader()
  
  projections = load_projected_representations(probe, model, dataloader)

  all_relations = []
  all_sentences = []
  all_idxs = []
  all_words = []
  all_diffs = []
  all_labels = []
  all_pairs = []
  i = 0
  for projection_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(projections, dataloader):
    for projection, label, length, (observation, _) in zip(projection_batch, label_batch, length_batch, observation_batch):
      for idx, word in enumerate(observation.sentence):
        if observation.head_indices[idx] == '0':
          pass # head word 
        else:
          head_index = int(observation.head_indices[idx])
          proj_diff = projection[idx] - projection[head_index-1]
          all_pairs.append((projection[idx], projection[head_index-1]))
          relation = observation.governance_relations[idx]
          all_diffs.append(proj_diff)
          all_sentences.append(" ".join(observation.sentence))
          all_idxs.append(idx)
          all_words.append(word)
          lang = [l for l in lang_mapping if i in lang_mapping[l]][0]
          all_labels.append(lang + '-' + relation)
      i += 1

  all_labels = np.array(all_labels)
  all_pairs = np.array(all_pairs)
  sentences_idxs_words = np.array([all_sentences, all_idxs, all_words]).T
  all_diffs = np.array(all_diffs) 
  np.save('/sailhome/ethanchi/structural-probes/relationOutputs/{}.npy'.format(output_name), all_diffs)
  np.save('/sailhome/ethanchi/structural-probes/relationOutputs/{}-pairs.npy'.format(output_name), all_pairs)
  np.save('/sailhome/ethanchi/structural-probes/relationOutputs/{}Y.npy'.format(output_name), all_labels)
  np.save('/sailhome/ethanchi/structural-probes/relationOutputs/{}-data.npy'.format(output_name), sentences_idxs_words)
  # write tsne
  if use_tsne:
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=229, verbose=10)
    print("Fitting TSNE.")
    # we want to take uniformly distributed samples from each language category:
    langs = lang_mapping.keys()
    langs_to_indices = {}
    total_wanted = 100000
    for lang in langs:
      selector = np.vectorize(lambda x: x.startswith(lang + '-'))
      indices_to_choose = np.where(selector(all_labels))[0]
      num_needed = total_wanted // len(langs)
      indices_needed = indices_to_choose[np.random.randint(low=0, high=indices_to_choose.shape[0], size=num_needed)]
      langs_to_indices[lang] = indices_needed
    indices = np.concatenate([langs_to_indices[lang] for lang in langs])
    # indices = np.random.randint(low=0, high=all_diffs.shape[0], size=40000)
    cut_diffs = all_diffs[indices]
    all_labels = all_labels.reshape(-1, 1)
    cut_labels = all_labels[indices]
    all_pairs = all_pairs.reshape(-1, 1)
    cut_pairs = cut_pairs[indices]
    print(sentences_idxs_words.dtype)
    cut_data = sentences_idxs_words[indices]
    try:
      projected = tsne.fit_transform(cut_diffs)
    except KeyboardInterrupt:
      pass
    print("Fitted.")
    print(cut_diffs.shape, cut_labels.shape, cut_data.shape)
    tsv_out = np.concatenate((projected, cut_labels, cut_data), axis=1)
    np.savetxt('/sailhome/ethanchi/structural-probes/relationOutputs/{}.tsv'.format(output_name), tsv_out, fmt="%s", header="x0\tx1\tlabel\tsentence\tidx\tword", comments="", delimiter="\t")
    np.save('/sailhome/ethanchi/structural-probes/relationOutputs/{}-cut-pairs.npy'.format(output_name), cut_pairs) 
    


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

  expt_dataset = dataset_class(args, task.DummyTask)
  expt_probe = probe_class(args)
  expt_model = model_class(args)

  evaluate_vectors(args, expt_probe, expt_dataset, expt_model, results_dir, output_name, use_tsne)


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

  # yaml_args = yaml.load(open(#")
  os.chdir(cli_args.dir)
  yaml_args = yaml.load(open(glob.glob("*.yaml")[0]))
  # setup_new_experiment_dir(cli_args, yaml_args, cli_args.results_dir)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  yaml_args['device'] = device
  execute_experiment(yaml_args, cli_args.dir, cli_args.output_name, cli_args.tsne)
