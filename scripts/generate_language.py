# every rule line should have the following form:
# the {N1} {V} the {N2}

# data should be one of the following three options:
# 1) single words [no characteristics]
# 2) list of forms of words, separated by slashes
# 3) list of words, followed by colon for characteristic

import re, itertools, sys, collections
import tqdm

get_base_token = lambda x: "".join(c for c in x if not c.isdigit()) 

def generateSentences(rule):
  tokens = re.findall("\{(.*?)\}", rule)
  assert(len(set(tokens)) == len(tokens))  # checking for uniqueness

  keywords = {}

  token_to_pos = collections.OrderedDict()
  has_characteristic = set()
  for token in tokens:
    token_to_pos[token] = [x for x in re.split('[{} ]', rule) if x != ''].index(token)
    token_base = get_base_token(token)
    if token_base not in keywords:
      with open(f'/u/scr/ethanchi/embeddings/{sys.argv[1]}/{token_base}.txt') as token_file:
        lines = [line.strip() for line in token_file if line != "\n"]
        assert(len(lines) > 0)
        if "/" in lines[0]:
          ets = [zip(line.split('/'), [str(x) for x in range(len(line.split('/')))]) for line in lines]
          ets = [word for sublist in ets for word in sublist]
          keywords[token_base] = ets
          has_characteristic.add(token_base)
        elif ":" in lines[0]:
          keywords[token_base] = [x.split(':') for x in lines]
          has_characteristic.add(token_base)
        else:
          keywords[token_base] = [(x,) for x in lines]
        
  swaps = []
  for token in tokens:
      swaps.append([(token, destination) for destination in keywords[get_base_token(token)]])
    
  # print(swaps)


  ## GENERATE LABELS
  label_line = ["sentence"]
  for token in tokens:
    label_line.append(token)
    label_line.append(token + "_idx")
    if get_base_token(token) in has_characteristic: label_line.append(token + "_char")

  out = ["\t".join(label_line)]

  for swap_product in tqdm.tqdm(itertools.product(*swaps)):
    # print(swap_product)
    output = rule
    total_spaces = 0
    for src, dest in swap_product:
      output = output.replace("{"f"{src}""}", dest[0])
      output += "\t" + str(dest[0])
      output += "\t" + str(token_to_pos[src] + total_spaces)
      total_spaces += dest[0].count(" ")
      if len(dest) > 1: output += "\t" + dest[1]
    out.append(output)
  return out

out = []
with open(f'/u/scr/ethanchi/embeddings/{sys.argv[1]}/template.txt', 'r') as template:
  for rule in template:
    out += generateSentences(rule.strip())
    
with open(f'/u/scr/ethanchi/embeddings/{sys.argv[1]}/sentences.txt', 'w') as out_file:
  out_file.write("\n".join(out))

# print("\n".join(out))

print(f"{len(out)} sentence{'' if len(out) == 1 else 's'} generated.")
