from src import connect, baseline, gnn_models, node2vec, ta_gcn
import sys
import warnings
warnings.filterwarnings("ignore")

def main():
    # 0. Collect command line arguments
    if len(sys.argv) != 2:
        sys.exit("Use: python run.py <dataset>")
    datasets = ['eth', 'test']
    dataset = sys.argv[1]
    if dataset not in datasets:
        sys.exit("wrong dataset name")

    # 1. Connect to TigerGraph database instance and upload graph schema
    # if dataset == 'test':
    #     connect.build_schema('test/testdata/test_nodes.csv', 'test/testdata/test_edges.csv')
    # else:
    #     connect.build_schema('data/nodes_train_test_split.csv', 'data/edges.csv')

    # 2. Add node features with GSQL queries
    # connect.add_features()

    # 3. Train models on transaction graph and evaluate performance

    # ################ Baseline Support Vector Machine #################
    df = connect.get_data('dataframe')
    # svm_acc = baseline.evaluate_baseline(df)
    # print("---------------------------------------------------------")
    # print("Baseline SVM Testing Accuracy: " + str(svm_acc))
    # print("---------------------------------------------------------")
    # ################## Avg Testing Accuracy: 60.5%  ##################


    # ##################### K-Nearest Neighbors ########################
    # knn_acc = baseline.evaluate_KNN(df)
    # print("---------------------------------------------------------")
    # print("Baseline KNN Testing Accuracy: " + str(knn_acc))
    # print("---------------------------------------------------------")
    # ################## Avg Testing Accuracy: 74.6%  #################


    ########################### XGBoost ##############################
    xgb_acc = baseline.evaluate_xgb(df)
    print("---------------------------------------------------------")
    print("Baseline XGBoost Testing Accuracy: " + str(xgb_acc))
    print("---------------------------------------------------------")
    ################## Avg Testing Accuracy: 81.6%  #################


    # ################### Graph Convolution Netowrk ####################
    # gnn_models.set_seed(123)
    # data = connect.get_data()
    # data, gcn, optimizer = gnn_models.initialize(data, 'gcn')
    # gnn_models.train(data, gcn, optimizer)
    # gcn_acc = gnn_models.evaluate(data, gcn)
    # print("---------------------------------------------------------")
    # print("GCN Testing Accuracy: " + str(gcn_acc))
    # print("---------------------------------------------------------")
    # ################## Avg Testing Accuracy: 79.6% ###################


    # ########################## GraphSAGE #############################
    # data, sage, optimizer = gnn_models.initialize(data, 'sage')
    # gnn_models.train(data, sage, optimizer)
    # gcn_acc = gnn_models.evaluate(data, sage)
    # print("---------------------------------------------------------")
    # print("GraphSAGE Testing Accuracy: " + str(gcn_acc))
    # print("---------------------------------------------------------")
    # ################## Avg Testing Accuracy: 81.9%  ##################


    # #################### Graph Attention Network #####################
    # data, gat, optimizer = gnn_models.initialize(data, 'gat')
    # gnn_models.train(data, gat, optimizer)
    # gat_acc = gnn_models.evaluate(data, gat)
    # print("---------------------------------------------------------")
    # print("GAT Testing Accuracy: " + str(gat_acc))
    # print("---------------------------------------------------------")
    # ################## Avg Testing Accuracy: 78.5%  ##################


    # ########################## Node2Vec ##############################
    # data, device, n2v, loader, optimizer = node2vec.initialize(data)
    # node2vec.train(device, n2v, loader, optimizer)
    # n2v_acc = node2vec.evaluate()
    # print("---------------------------------------------------------")
    # print("Node2Vec Testing Accuracy: " + str(n2v_acc))
    # print("---------------------------------------------------------")
    # ################## Avg Testing Accuracy: 76.6%  ##################


    # ########################### TA-GCN ###############################
    # data, model, optimizer = ta_gcn.initialize(data)
    # ta_gcn_acc = ta_gcn.evaluate(data, model, optimizer)
    # print("---------------------------------------------------------")
    # print("TA-GCN Testing Accuracy: " + str(ta_gcn_acc))
    # print("---------------------------------------------------------")
    # ################## Avg Testing Accuracy: 82.2%  ##################

    return

if __name__ == "__main__":
    main()