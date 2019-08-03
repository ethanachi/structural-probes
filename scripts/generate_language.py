# every rule line should have the following form:
# the {N1} {V} the {N2}

# data should be one of the following three options:
# 1) single words [no characteristics]
# 2) list of forms of words, separated by slashes
# 3) list of words, followed by colon for characteristic

import re, itertools

def generateSentences(rule):
  tokens = re.findall("\{(.*?)\}", text)
  assert(len(set(tokens)) == len(tokens))  # checking for uniqueness

  keywords = []
  
  for token in tokens:
    token_base = "".join(c for c in token if not c.isdigit())
    with open(f'/u/scr/ethanchi/embeddings/{sys.argv[1]}/{token_base}.txt') as token_file:
      lines = [line.strip() for line in token_file if line != "\n"]
      assert(len(lines) > 0)
      if "/" in lines[0]:
        ets = [zip(range(len(line.split('/'))), [line.split('/')]) for line in lines]
        ets = [word for sublist in ets for word in sublist]
        keywords[token] = ets
      elif ":" in lines[0]:
        ets = [x.split(':') for x in lines]
      else:
        ets = [(x,) for x in lines]
  
  swaps = []
  for token in tokens:
      swaps.append([(token, destination) for destination in keywords[token]])
    
  print(swaps)

  for swap_product in itertools.product(*swaps):
    output = rule
    to_add = []
    for src, dest in swap_product:
      output = text.replace(f"{src}", dest[0])
      if len(dest) > 1: to_add.append(dest[1])
    print(output + "\t" + "\t".join(to_add))

out = []
with open(f'/u/scr/ethanchi/embeddings/{sys.argv[1]}/template.txt', 'r') as template:
  for rule in template:
    "".append(generateSentences(rule))
    
with open(f'/u/scr/ethanchi/embeddings/{sys.argv[1]}/sentences.txt', 'w') as out_file:
  out_file.write("\n".join(out))

print("\n".join(out))

print(f"{len(out)} sentence{'' if len(out != 1 else 's')} generated.")
