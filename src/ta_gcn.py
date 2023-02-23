import torch
import torch.nn.functional as F
from torch_geometric.nn import TAGConv

class TA_GCN(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        self.conv1 = TAGConv(11, 128, K=2, aggr='min')
        self.conv2 = TAGConv(128, 2, K=2, aggr='min')
        self.data = data

    def forward(self, x, edge_index):
        x, edge_index = self.data.x.float(), self.data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def initialize(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = TA_GCN(data).to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    return data, model, optimizer

def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    inp = model(data.x, data.edge_index)[data.is_train]
    out = data.y[data.is_train].long()
    F.nll_loss(inp, out).backward()
    optimizer.step()

@torch.no_grad()
def test(data, model):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('is_train','is_test'):
        pred = out[mask].argmax(1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def evaluate(data, model, optimizer):
    for epoch in range(1, 101):
        train(data, model, optimizer)
        train_acc, test_acc = test(data, model)
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    final_acc = test(data, model)[1]
    return final_acc