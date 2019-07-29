import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

data = np.load("/sailhome/ethanchi/structural-probes/relationOutputs/allRelations.npy") 

labels = np.load("/sailhome/ethanchi/structural-probes/relationOutputs/allRelationsY.npy")
print(np.unique(labels).shape)
print("Finished loading data.")
print(labels)

colors = ['xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:pink', 'xkcd:brown', 'xkcd:red', 'xkcd:light blue', 'xkcd:teal', 'xkcd:orange', 'xkcd:light green', 'xkcd:magenta', 'xkcd:yellow', 'xkcd:sky blue', 'xkcd:grey', 'xkcd:lime green', 'xkcd:light purple', 'xkcd:violet', 'xkcd:dark green', 'xkcd:turquoise', 'xkcd:lavender', 'xkcd:dark blue', 'xkcd:tan', 'xkcd:cyan', 'xkcd:aqua', 'xkcd:forest green', 'xkcd:mauve', 'xkcd:dark purple', 'xkcd:bright green', 'xkcd:maroon', 'xkcd:olive', 'xkcd:salmon', 'xkcd:beige', 'xkcd:royal blue', 'xkcd:navy blue', 'xkcd:lilac', 'xkcd:black', 'xkcd:hot pink', 'xkcd:light brown', 'xkcd:pale green', 'xkcd:peach', 'xkcd:olive green', 'xkcd:dark pink']

try:
    tsne = TSNE(n_components=2, random_state=229, verbose=10)
except KeyboardInterrupt:
    print("Interrupted.")
    exit()
print("Fitting.")
projected = tsne.fit_transform(data)
print("Fitted.")

plt.figure(figsize=(12, 10))
for i, c in zip(list(np.unique(labels)), colors):
    plt.scatter(projected[labels == i, 0], projected[labels == i, 1], c=c, label=i)
plt.legend()
plt.savefig('/sailhome/ethanchi/structural-probes/relationOutputs/fig.png')



