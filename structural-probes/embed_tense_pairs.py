import pytorch

embeddings = np.load(f'/u/scr/ethanchi/embeddings/{sys.argv[1]}/embeddings.npy')
embeddings = np.stack((embeddings[::2], embeddings[1::2]), axis=1)

embeddings = embeddings[:, 0] - embeddings[:, 1]
print(np.norm(embeddings.mean()))




