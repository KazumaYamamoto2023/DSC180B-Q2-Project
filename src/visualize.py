import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.convert import to_networkx
from torch_geometric.explain import Explainer, GNNExplainer, ExplainerConfig, ModelConfig
from connect import get_data


def model_comparison():
    """
    Helper function to compare performance metrics of
    baseline classifiers to graph neural networks
    """
    data = {"Model":['SVM', "kNN", "XGB", 'GCN', "GAT", "SAGE", "N2V", "TAGCN"],
        "Avg Testing Accuracy":[60.5,74.6,81.6,79.6,78.5,81.9,76.6,82.2],
        "Type":['Non-Graph', 'Non-Graph', 'Non-Graph', 'GNN', 'GNN', 'GNN', 'GNN', 'GNN']}
    df = pd.DataFrame(data).sort_values(by="Avg Testing Accuracy", ascending=False)
    return df


def sample_vertex():
    """
    Helper function to randomly sample the index of
    a fraudulent wallet in the transaction network.
    """
    data = get_data(type='dataframe')
    fraud_idx = data[data['label']==1].sample(1)
    return int(fraud_idx.index[0])


def explainability(data, model, node_idx):
    """
    Helper function to create a model explanation object
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        model_config=ModelConfig(
            mode='classification',
            task_level='node',
            return_type='log_probs',
        ),
        explainer_config=ExplainerConfig(
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
        )
    )
    node_index = node_idx
    explanation = explainer(data.x.float(), data.edge_index, index=node_index)
    print(f'Generated explanations in {explanation.available_explanations}')
    return explanation


def feature_importance(explanation, outpath):
    """
    Helper function to determine feature importance
    from predictor variables
    """
    node_mask = explanation.node_feat_mask
    feat_importance = node_mask.sum(dim=0).cpu().numpy()
    feat_labels = ["in_degree","out_degree","total_sent","min_sent","max_sent","avg_sent","total_recv","min_recv","max_recv","avg_recv","pagerank"]

    df = pd.DataFrame({'feat_importance': feat_importance}, index=feat_labels)
    df = df.sort_values("feat_importance", ascending=False)
    df = df.round(decimals=3)

    ax = df.plot(
        kind='barh',
        figsize=(10, 7),
        title="Node Feature Importance",
        ylabel='Node Feature',
        xlim=[0, float(feat_importance.max()) + 0.3],
        legend=False,
    )
    plt.gca().invert_yaxis()
    ax.bar_label(container=ax.containers[0], label_type='edge')
    plt.savefig(outpath)
    plt.show()


def visualize_subgraph(explanation, data, node_idx, outpath):
    """
    Helper function to visualize transaction network subgraph
    of a wallet classified as 'fraudulent'
    """
    node_idx = node_idx
    subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
        node_idx, 2, data.edge_index, relabel_nodes=True,
        num_nodes=None)

    edge_mask = explanation.edge_mask[hard_edge_mask]
    y = data.y[subset].to(torch.float) / data.y.max().item()

    edge_color = ['black'] * edge_index.size(1)
    data = Data(edge_index=edge_index, att=edge_mask,
                edge_color=edge_color, y=y, num_nodes=y.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'],
                    edge_attrs=['att', 'edge_color'])
    mapping = {k: i for k, i in enumerate(subset.tolist())}
    G = nx.relabel_nodes(G, mapping)

    pos = nx.spring_layout(G, seed=123)
    ax = plt.gca()
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->",
                alpha=max(data['att'], 0.5),
                color=data['edge_color'],
                shrinkA=np.sqrt(800) / 2.0,
                shrinkB=np.sqrt(800) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ))

    colors = ['red' if label==1 else 'green' for label in y.tolist()]
    nx.draw_networkx_nodes(G, pos, node_color=colors)
    nx.draw_networkx_labels(G, pos)
    plt.title("Subgraph of Random Fraudulent Wallet")
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Fraud',
                       markerfacecolor='r', markersize=15),
                      Line2D([0], [0], marker='o', color='w', label='Non-fraud',
                       markerfacecolor='g', markersize=15)]
    plt.legend(handles=legend_elements, loc='best')
    plt.savefig(outpath)
    return ax, G