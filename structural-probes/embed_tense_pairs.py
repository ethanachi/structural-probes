import numpy as np
import torch
import torch.nn as nn
import sys

USE_PROJECTED = "--projected" in sys.argv 
pathword = "representations" if USE_PROJECTED else "embeddings"
    
embeddings = np.load(f'/u/scr/ethanchi/embeddings/{sys.argv[1]}/{pathword}.npy')
print(embeddings.shape)
input_size = embeddings.shape[-1]
embeddings = np.stack((embeddings[::2], embeddings[1::2]), axis=1)
print(embeddings.shape)
embeddings = embeddings[:, :, 2] # take the verb
print(embeddings.shape)
embeddings = embeddings.reshape(67, -1, 2, input_size)
print(embeddings.shape)

num_verbs = embeddings.shape[0]
ordering = np.arange(num_verbs)
np.random.shuffle(ordering)
split = int(num_verbs * .9)

train_embeddings = embeddings[ordering[:split]]

train_embeddings_flat = train_embeddings.reshape(-1, 2, input_size)

test_embeddings = embeddings[ordering[split:]]
test_embeddings_flat = test_embeddings.reshape(-1, 2, input_size)

train_set_unnorm = train_embeddings_flat[:, 0] - train_embeddings_flat[:, 1]
train_set = train_set_unnorm / np.linalg.norm(train_set_unnorm, axis=1, keepdims=True)
test_set_unnorm = test_embeddings_flat[:, 0] - test_embeddings_flat[:, 1]
test_set = test_set_unnorm / np.linalg.norm(test_set_unnorm, axis=1, keepdims=True)

print(train_set.shape, test_set.shape)

class DirectionProbe(nn.Module):
    def __init__(self, inputSize):
        super(DirectionProbe, self).__init__()
        self.inputSize = inputSize
        self.u = nn.Parameter(data = torch.zeros(self.inputSize))
        nn.init.uniform_(self.u, -0.05, 0.05)
    def forward(self, batch):
        return torch.mean(batch @ (self.u / torch.norm(self.u)))

train_inputs = torch.Tensor(train_set)
test_inputs = torch.Tensor(test_set)
model = DirectionProbe(input_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

NUM_EPOCHS = 2000

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    output = model(train_inputs)
    error = 1 - output
    error.backward()
    optimizer.step()
    if epoch % 100 == 0:
        test_output = model(test_inputs)
        print("Epoch {}, train score {}, test score {}".format(epoch, output, test_output))

direction_vector = model.u.cpu().detach().numpy()
np.save(f"/u/scr/ethanchi/embeddings/{sys.argv[1]}/{pathword}-vector.npy", direction_vector)
direction_vector /= np.linalg.norm(direction_vector)
print(np.linalg.norm(direction_vector))

EPSILON = 0.0001

## EVALUATION

numSuccessful = len(np.where(np.linalg.norm(test_set_unnorm + EPSILON * direction_vector, axis=1) > np.linalg.norm(test_set_unnorm, axis=1))[0])

print("{:.2f}% accuracy".format(numSuccessful / test_embeddings_flat.shape[0] * 100))

angle = np.dot(test_set, direction_vector[..., np.newaxis])
print(angle.shape)
print(angle.mean())

NUM_TRIALS = 2000
accuracies = 0.0
angles = 0.0
for i in range(NUM_TRIALS):
    rand_direction_vector = np.random.randn(input_size)
    rand_direction_vector /= np.linalg.norm(rand_direction_vector)
    numSuccessful = len(np.where(np.linalg.norm(test_set_unnorm + EPSILON * rand_direction_vector, axis=1) > np.linalg.norm(test_set_unnorm, axis=1))[0])
    accuracies += numSuccessful / test_embeddings_flat.shape[0]
    angles += np.dot(test_set, rand_direction_vector[..., np.newaxis]).mean()

print("{:.2f}% accuracy".format(accuracies / NUM_TRIALS * 100))
print("{:.2f} angle".format(angles / NUM_TRIALS))
