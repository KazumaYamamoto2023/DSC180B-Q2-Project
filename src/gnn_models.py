from pyTigerGraph.gds.metrics import Accumulator, Accuracy
import pyTigerGraph as tg
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, GCN, GAT


def initialize(data, model_selection):
    """
    Helper function to initialize a 'gcn', 'sage', or 'gat' model
    from pytorch.
    """
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    if model_selection == 'gcn':
        # set hyperparameters
        hp = {"hidden_dim": 128, "num_layers": 2, "dropout": 0.05, "lr": 0.0075, "l2_penalty": 5e-5}
        # initialize model
        model = GCN(
            in_channels=7,
            hidden_channels=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            out_channels=7,
            dropout=hp["dropout"],
            heads=8
        ).to(device)

    elif model_selection == 'sage':
        # set hyperparameters
        hp = {"hidden_dim": 128, "num_layers": 2, "dropout": 0.05, "lr": 0.0075, "l2_penalty": 5e-5}
        # initialize model
        model = GAT(
            in_channels=7,
            hidden_channels=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            out_channels=7,
            dropout=hp["dropout"],
            heads=8
        ).to(device)

    else:
        # set hyperparameters
        hp = {"hidden_dim": 128, "num_layers": 2, "dropout": 0.05, "lr": 0.0075, "l2_penalty": 5e-5}
        # initialize model
        model = GraphSAGE(
            in_channels=7,
            hidden_channels=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            out_channels=7,
            dropout=hp["dropout"],
            heads=8
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["l2_penalty"]
    )
    return data, model, optimizer


def train(data, model, optimizer):
    logs = {}
    for epoch in range(30):
        # Train
        model.train()
        acc = Accuracy()
        # Forward pass
        out = model(data.x.float(), data.edge_index)
        # Calculate loss
        loss = F.cross_entropy(out[data.is_train].float(), data.y[data.is_train].long())
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Evaluate
        val_acc = Accuracy()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            acc.update(pred[data.is_train], data.y[data.is_train])
            valid_loss = F.cross_entropy(out[data.is_test].float(), data.y[data.is_test].long())
            val_acc.update(pred[data.is_test], data.y[data.is_test])
        # Logging
        logs["loss"] = loss.item()
        logs["test_loss"] = valid_loss.item()
        logs["acc"] = acc.value
        logs["test_acc"] = val_acc.value
        print(
            "Epoch: {:02d}, Train Loss: {:.4f}, Test Loss: {:.4f}, Train Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(
                epoch, logs["loss"], logs["test_loss"], logs["acc"], logs["test_acc"]
            )
        )

def evaluate(data, model):
    model.eval()
    acc = Accuracy()
    with torch.no_grad():
        pred = model(data.x.float(), data.edge_index).argmax(dim=1)
        acc.update(pred[data.is_test], data.y[data.is_test])
    return acc.value