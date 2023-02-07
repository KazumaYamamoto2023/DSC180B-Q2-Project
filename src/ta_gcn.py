import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import TAGConv

class TA_GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TAGConv(7, 16)
        self.conv2 = TAGConv(16, 2)

    def forward(self):
        x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def intialize(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = TA_GCN().to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    return data, model, optimizer

def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    inp = model()[data.is_train]
    out = data.y[data.is_train].long()
    F.nll_loss(inp, out).backward()
    optimizer.step()

@torch.no_grad()
def test(data, model):
    model.eval()
    out, accs = model(), []
    for _, mask in data('is_train','is_test'):
        pred = out[mask].argmax(1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def evaluate(data, model, optimizer):
    best_acc = 0
    for epoch in range(1, 201):
        train(data, model, optimizer)
        train_acc, tmp_test_acc = test(data, model)
        if tmp_test_acc > best_acc:
            best_acc = tmp_test_acc
            test_acc = tmp_test_acc
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    return best_acc

# testing accuracy roughly 74% (best)