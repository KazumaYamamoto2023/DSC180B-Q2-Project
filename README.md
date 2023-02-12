# Graph-Based Deep Learning for Fraud Detection in Ethereum Transaction Networks
This project aims to compare graph-based to non-graph based algorithms for fraud detection in Ethereum transaction networks. We will predict whether a given Ethereum wallet in the transaction graph is fraudulent or non-fraudulent, given the wallet's transaction history in the network.

Graph exploration, analysis, and model building will be conducted using [TigerGraph](https://tgcloud.io/), an enterprise-scale graph data platform for advanced analytics and machine learning. 

Model performance was determined by taking the average classification accuracy on the testing set over 10 model runs. The resulting classifier performance for this prediction task are as follows:

* Support Vector Machine (~60.5%)
* Graph Convolutional Network (~79.6%)
* Graph Attention Network (~78.5%)
* GraphSAGE (~81.9%)
* Node2Vec (~76.6%)
* Topology Adaptive Graph Convolutional Network (~82.2%)

## Getting Started
1. Create a free [TigerGraph](https://tgcloud.io/) account and launch a free cluster. Save the cluster's domain name in `config/tigergraph.json`.

2. Open [GraphStudio](https://tgcloud.io/app/tools/GraphStudio/) and create a new graph named 'Ethereum'

3. Open [AdminPortal](https://tgcloud.io/app/tools/Admin%20Portal/) and navigate to the "Management" tab and select "Users." Generate a secret alias and secret value, and save the secret value in `config/tigergraph.json`.

4. Run `python run.py eth` to connect to the TigerGraph database instance, build the graph schema, load the dataset, and evaluate the models

    * This process is detailed in `notebooks/tg_data_loading.ipynb`
    * Run `python run.py test` to evaluate the models on a subset of the Ethereum transaction network


## Data Description
This dataset contains transaction records of 445 phishing accounts and 445 non-phishing accounts of Ethereum. We obtain 445 phishing accounts labeled by [Etherscan](etherscan.io) and the same number of randomly selected unlabeled accounts as our objective nodes. The dataset can be used to conduct node classification of financial transaction networks. 

We collect the transaction records based on an assumption that for a typical money transfer flow centered on a phishing node, the previous node of the phishing node may be a victim, and the next one to three nodes may be the bridge nodes with money laundering behaviors, as figure shows. Therefore, we collect subgraphs by [K-order sampling](https://ieeexplore.ieee.org/document/8964468) with K-in = 1, K-out = 3 for each of the 890 objective nodes and then splice them into a large-scale network with 86,623 nodes. 

![A schematic illustration of a directed K-order subgraph for phishing node classification.](https://s1.ax1x.com/2020/03/27/GCZGmd.md.jpg)

## Data Source
[XBlock](http://xblock.pro/#/dataset/6) collects the current mainstream blockchain data and is one of the blockchain data platforms with the largest amount of data and the widest coverage in the academic community.
```
@article{ wu2019tedge,
  author = "Jiajing Wu and Dan Lin and Qi Yuan and Zibin Zheng",
  title = "T-EDGE: Temporal WEighted MultiDiGraph Embedding for Ethereum Transaction Network Analysis",
  journal = "arXiv preprint arXiv:1905.08038",
  year = "2019",
  URL = "https://arxiv.org/abs/1905.08038"
}
```