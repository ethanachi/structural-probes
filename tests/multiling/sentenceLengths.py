import sys
mapping = {}
total = 0
with open(sys.argv[1], 'r') as f:
    for line in f:
        total += 1
        l = len(line.split())
        mapping[l] = mapping.get(l, 0) + 1

print(mapping)
percentiles = {}
sumSoFar = 0
for l in sorted(mapping.keys()):
    sumSoFar += mapping[l]
    print("{}\t{:.3f}%".format(l, sumSoFar/total*100))
