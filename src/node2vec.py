import sys
import torch
from torch_geometric.nn import Node2Vec

def initialize(data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=64, walk_length=10,
                    context_size=5, walks_per_node=5,
                    num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    num_workers = 0 if sys.platform.startswith('win') else 4
    loader = model.loader(batch_size=64, shuffle=True,
                        num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    return device, model, loader, optimizer

def train(device, model, loader, optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test(data, model):
    model.eval()
    z = model()
    acc = model.test(z[data.is_train], data.y[data.is_train],
                     z[data.is_test], data.y[data.is_test],
                     max_iter=150)
    return acc

def evaluate():
    for epoch in range(1, 101):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:02d}, Train Loss: {loss:.4f}, Test Acc: {acc:.4f}')
    final_acc = test()
    return final_acc
    