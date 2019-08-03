import numpy as np
import torch
import torch.nn as nn
import sys

embeddings = np.load(f'/u/scr/ethanchi/embeddings/{sys.argv[1]}/embeddings.npy')
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

train_embeddings = embeddings[ordering[:split]].reshape(-1, 2, input_size)
test_embeddings = embeddings[ordering[split:]].reshape(-1, 2, input_size)

train_set = train_embeddings[:, 0] - train_embeddings[:, 1]
train_set = train_set / torch.norm(train_set, dim=1, keepdim=True)
test_set = test_embeddings[:, 0] - test_embeddings[:, 1]
test_set = train_set / torch.norm(test_set, dim=1, keepdim=True)

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

NUM_EPOCHS = 5000

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
EPSILON = 0.00001
embeddings
