""" Demos a trained structural probe by making parsing predictions on stdin """

from argparse import ArgumentParser
import os
from datetime import datetime
import shutil
import yaml
from tqdm import tqdm
import torch
import numpy as np
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
mpl.rcParams['agg.path.chunksize'] = 10000

import data
import model
import probe
import regimen
import reporter
import task
import loss
import run_experiment

from pytorch_pretrained_bert import BertTokenizer, BertModel




tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertModel.from_pretrained('bert-multilingual-cased')
LAYER_COUNT = 24
FEATURE_COUNT = 1024
MAX_SENTENCE_LENGTH = 25
model.to(args['device'])
model.eval()

probe = probe.TwoWordPSDProbe(args)
probe.load_state_dict(torch.load(args['probe']['distance_params_path'], map_location=args['device']))
probe.eval()
probe.proj()


outputs = []
sentences = []

for index, line in tqdm(enumerate(sys.stdin), desc='[projecting]'):
  # Tokenize the sentence and create tensor inputs to BERT
  untokenized_sent = line.strip().split()
  if len(untokenized_sent) > MAX_SENTENCE_LENGTH:
    raise(AssertionError)
  tokenized_sent = tokenizer.wordpiece_tokenizer.tokenize('[CLS] ' + ' '.join(line.strip().split()) + ' [SEP]')
  untok_tok_mapping = data.SubwordDataset.match_tokenized_to_untokenized(tokenized_sent, untokenized_sent)

  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
  segment_ids = [1 for x in tokenized_sent]

  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segment_ids])

  tokens_tensor = tokens_tensor.to(args['device'])
  segments_tensors = segments_tensors.to(args['device'])

  with torch.no_grad():
    # Run sentence tensor through BERT after averaging subwords for each token
    encoded_layers, _ = model(tokens_tensor, segments_tensors)
    single_layer_features = encoded_layers[args['model']['model_layer']]
    representation = torch.stack([torch.mean(single_layer_features[0,untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], dim=0) for i in range(len(untokenized_sent))], dim=0)

    # Run BERT token vectors through the trained probes
    projected = torch.matmul(representation, probe.proj)
    projected = torch.nn.functional.pad(projected, (0, 0, 0, 25-projected.shape[1]), mode="constant", value=0)
    print(projected.shape)
    outputs.append(projected)
    sentences.append(line.detach().cpu().numpy())
    
out = torch.Tensor(sentences)
print(out.shape)
    
      

if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('experiment_config')
  argp.add_argument('--results-dir', default='',
      help='Set to reuse an old results dir; '
      'if left empty, new directory is created')
  argp.add_argument('--seed', default=0, type=int,
      help='sets all random seeds for (within-machine) reproducibility')
  cli_args = argp.parse_args()
  if cli_args.seed:
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  yaml_args= yaml.load(open(cli_args.experiment_config))
  run_experiment.setup_new_experiment_dir(cli_args, yaml_args, cli_args.results_dir)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  yaml_args['device'] = device
  report_on_stdin(yaml_args)
