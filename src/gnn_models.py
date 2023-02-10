import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, GCN, GAT
from pyTigerGraph.gds.metrics import Accuracy

def set_seed(seed):
    """
    Helper function to set the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def initialize(data, model_selection):
    """
    Helper function to initialize a 'gcn', 'sage', or 'gat' model
    from pytorch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    if model_selection == 'gcn':
        # set hyperparameters
        hp = {"in_dim": 11, "hidden_dim": 64, "num_layers": 3, "dropout": 0.05, "lr": 0.05, "l2_penalty": 5e-5}
        # initialize model
        model = GCN(
            in_channels=hp["in_dim"],
            hidden_channels=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            out_channels=hp["in_dim"],
            dropout=hp["dropout"],
        ).to(device)

    elif model_selection == 'sage':
        # set hyperparameters
        hp = {"in_dim": 11, "hidden_dim": 128, "num_layers": 2, "lr": 0.05, "l2_penalty": 5e-5}
        # initialize model
        model = GraphSAGE(
            in_channels=hp["in_dim"],
            hidden_channels=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            out_channels=hp["in_dim"],
        ).to(device)

    else:
        # set hyperparameters
        hp = {"in_dim": 11, "hidden_dim": 32, "num_layers": 3, "lr": 0.005, "l2_penalty": 5e-5}
        # initialize model
        model = GAT(
            in_channels=hp["in_dim"],
            hidden_channels=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            out_channels=hp["in_dim"],
            heads=8
        ).to(device)
    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["l2_penalty"]
    )
    return data, model, optimizer


def train(data, model, optimizer):
    logs = {}
    for epoch in range(100):
        # train model
        model.train()
        acc = Accuracy()
        # forward pass
        out = model(data.x.float(), data.edge_index)
        # compute loss
        loss = F.cross_entropy(out[data.is_train].float(), data.y[data.is_train].long())
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # evaluate model on test set
        val_acc = Accuracy()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            acc.update(pred[data.is_train], data.y[data.is_train])
            valid_loss = F.cross_entropy(out[data.is_test].float(), data.y[data.is_test].long())
            val_acc.update(pred[data.is_test], data.y[data.is_test])
        # logs
        logs["loss"] = loss.item()
        logs["test_loss"] = valid_loss.item()
        logs["acc"] = acc.value
        logs["test_acc"] = val_acc.value
        print("Epoch: {:02d}, Train Loss: {:.4f}, Test Loss: {:.4f}, Train Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(
                epoch, logs["loss"], logs["test_loss"], logs["acc"], logs["test_acc"]))

def evaluate(data, model):
    model.eval()
    acc = Accuracy()
    with torch.no_grad():
        pred = model(data.x.float(), data.edge_index).argmax(dim=1)
        acc.update(pred[data.is_test], data.y[data.is_test])
    return acc.value