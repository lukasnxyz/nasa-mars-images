import csv
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

d_opts = [('cuda', torch.cuda.is_available()), ('mps', torch.backends.mps.is_available()), ('cpu', True)]
device = next(device for device, available in d_opts if available)
print(f'using device: {device}')

data_csv = 'data/nasa_mars_images.csv'
with open(data_csv, mode='r') as f:
    reader = csv.reader(f)
    data = np.array([[np.float32(item) for item in row] for row in tqdm(reader, desc='loading csv data')])
    
n = int(0.85*len(data))
data_tr, data_val = data[:n], data[n:]
Xtr, Ytr = torch.from_numpy(data_tr[:, 1:]).to(device), torch.from_numpy(data_tr[:, 0]).to(device)
Xval, Yval = torch.from_numpy(data_val[:, 1:]).to(device), torch.from_numpy(data_val[:, 0]).to(device)

torch.manual_seed(42)
epochs = 10000
epoch_itr = 1000
batch_size = 32
n_hidden_units = 128
learning_rate = 1e-3
dropout = 0.2

class Model_v1(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=n_in, out_channels=n_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=n_hidden, out_channels=n_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Linear(n_hidden, n_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.block3(self.block2(self.block1(x)))
    
model = Model_v1(n_in=Xtr.shape[1], n_hidden=n_hidden_units, n_out=8).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model.train()
start_time = time.time()
for epoch in tqdm(range(epochs)):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model.forward(Xb)
    loss = F.cross_entropy(logits, Yb)

    optimizer.zero_grad()
    loss.backward()

    learning_rate = 1e-3 if epoch < 5000 else 1e-4
    optimizer.step()

    if epoch % epoch_itr == 0:
        print(f'{epoch}: loss {loss.item():.4f}')  
end_time = time.time()
print(f'time to train {end_time - start_time:.1f}s')

@torch.no_grad()
def split_loss(split: str):
    x,y = {
        'train': (Xtr, Ytr),
        'val': (Xval, Yval),
    }[split]
    logits = model(x.float())
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

with torch.inference_mode():
    split_loss('train')
    split_loss('val')
  
model.eval()
with torch.inference_mode():
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc
    logits = model(Xval)
    acc = accuracy_fn(y_pred=logits.argmax(dim=1),
                     y_true=Yval)
    print(f'Accuracy: {acc:.2f}%')

# 1.
# val loss: 1.4578
# acc: 81.65%

# 2.