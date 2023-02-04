from src import connect, train

def main():
    # 1. Connect to TigerGraph database instance and upload graph schema
    connect.build_schema()
    # 2. Add node features with GSQL queries
    connect.add_features()
    # 3. Train models on transaction graph
    train.train_baseline()
    train.train_graphsage()
    train.train_node2vec()
    train.train_gcn()
    train.train_pcgnn()
    # 4. Evaluate different models
    evaluate()
    return

if __name__ == "__main__":
    main()