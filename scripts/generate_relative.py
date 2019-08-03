import sys

with open(f"/u/scr/ethanchi/embeddings/{sys.argv[1]}/verbs.txt", "r") as verb_file:
    verbs = [x.strip() for x in verb_file.readlines() if x != "\n"]
    verbs = [x.split('/') for x in verbs]

with open(f"/u/scr/ethanchi/embeddings/{sys.argv[1]}/nouns.txt", "r") as noun_file:
    nouns = [x.strip() for x in noun_file.readlines() if x != "\n"]

# print(verbs)

for verb in verbs:
    for noun in nouns:
        for noun2 in nouns:
            for tense in range(len(verb)):
                print(f"Le {noun} {verb[tense]} le {noun2}", end="\t")
                print(("present", "past", "future")[tense])


