from src import connect, baseline, ta_gcn, gnn_models
import warnings
warnings.filterwarnings("ignore")

def main():
    # 1. Connect to TigerGraph database instance and upload graph schema
    connect.build_schema()

    # 2. Add node features with GSQL queries
    connect.add_features()

    # 3. Train models on transaction graph and evaluate performance

    ### Baseline Support Vector Machine###
    df = connect.get_data('dataframe')
    svm_acc = baseline.evaluate_baseline(df)
    print("Baseline SVM Testing Accuracy: " + str(svm_acc))
    print("-----------------------------------------")



    ### Graph Convolution Netowrk ###
    data, sage, optimizer = gnn_models.initialize(data, 'sage', optimizer)
    gnn_models.train(data, sage, optimizer)
    sage_acc = gnn_models.evaluate(data, sage)
    print("GraphSAGE Testing Accuracy: " + str(sage_acc))
    print("-----------------------------------------")

    ### GraphSAGE ###
    data = connect.get_data()
    data, gcn, optimizer = gnn_models.initialize(data, 'gcn', optimizer)
    gnn_models.train(data, gcn, optimizer)
    gcn_acc = gnn_models.evaluate(data, gcn)
    print("GCN Testing Accuracy: " + str(gcn_acc))
    print("-----------------------------------------")

    ### Graph Attention Network ###
    data, gat, optimizer = gnn_models.initialize(data, 'gat', optimizer)
    gnn_models.train(data, gat, optimizer)
    gat_acc = gnn_models.evaluate(data, gat)
    print("GraphSAGE Testing Accuracy: " + str(gat_acc))
    print("-----------------------------------------")

    # Node2Vec?? 

    ### TA-GCN ###
    data, model, optimizer = ta_gcn.initialize(data)
    ta_gcn_acc = ta_gcn.evaluate(data, model, optimizer)
    print("TA-GCN Testing Accuracy: " + ta_gcn_acc)
    print("-----------------------------------------")

    # model validation? tsne visual?

    return

if __name__ == "__main__":
    main()