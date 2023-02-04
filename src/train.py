# train models
import pyTigerGraph as tg
import json

# define hyperparameters for model training
hp = {"batch_size": 5000,
    "num_neighbors": 200,
    "num_hops": 3,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.05,
    "lr": 0.0075,
    "l2_penalty": 5e-5}

def get_graph():
    """
    A function to connect to the TigerGraph instance, split verticies into train/test/validation sets
    and returns a graph loader object containing the dataset stored in TigerGraph.
    """
    # import TigerGraph instance config
    with open('config/tigergraph.json', 'r') as f:
        config = json.load(f)

    # Connection parameters
    hostName = config['host']
    secret = config['secret']
    conn = tg.TigerGraphConnection(host=hostName, gsqlSecret=secret, graphname="Ethereum")
    conn.getToken(secret)

    # split verticies into train/test/validation sets
    split = conn.gds.vertexSplitter(is_training=0.8, is_testing=0.1, is_validation=0.1)
    split.run()

    # load training data from database
    graph_loader = conn.gds.neighborLoader(
        v_in_feats=["in_degree","out_degree","send_amount","send_min","recv_amount","recv_min","pagerank"],
        v_out_labels=["is_fraud"],
        v_extra_feats=["is_training", "is_validation"],
        output_format="PyG",
        batch_size=hp["batch_size"],
        num_neighbors=hp["num_neighbors"],
        num_hops=hp["num_hops"],
        filter_by = "is_training",
        shuffle=True)
    
    return graph_loader


def train_baseline():
    return

def train_graphsage():
    return

def train_node2vec():
    return

def train_gcn():
    global_steps = 0
    logs = {}
    for epoch in range(10):
        # Train
        print("Start Training epoch:", epoch)
        model.train()
        epoch_train_loss = Accumulator()
        
        epoch_train_auc = []
        epoch_train_prec = []
        epoch_train_rec = []
        epoch_train_apr = []
        epoch_best_thr = []
        
        # Iterate through the loader to get a stream of subgraphs instead of the whole graph
        for bid, batch in enumerate(train_loader):
            # print(bid, batch)
            batchsize = batch.x.shape[0]
            norm = T.NormalizeFeatures()
            batch = norm(batch).to(device)
            batch.x = batch.x.type(torch.FloatTensor)
            batch.y = batch.y.type(torch.LongTensor)
        
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.edge_weight)
            # Calculate loss
            class_weight = torch.FloatTensor([1.0, 15.0])
            loss = F.cross_entropy(out[batch.is_training], batch.y[batch.is_training], class_weight)
            # f1_loss(batch.y[batch.is_training], out[batch.is_training], is_training=True)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss.update(loss.item() * batchsize, batchsize)
            # Predict on training data
            with torch.no_grad():
                pred = out.argmax(dim=1)
                y_pred = out[batch.is_training][:,1].cpu().numpy()
                y_true = batch.y[batch.is_training].cpu().numpy()
                # softmax = F.softmax(out, dim=1)[batch.is_training][:,1].cpu().numpy()
                best_threshold = threshold_search(y_true, y_pred)
                y_pred1 = (y_pred > best_threshold).astype(int)

                try:
                    epoch_train_auc.append(roc_auc_score(y_true, y_pred))
                except:
                    epoch_train_auc.append(np.NaN)
                    
                epoch_train_prec.append(precision_score(y_true, y_pred1))
                epoch_train_rec.append(recall_score(y_true, y_pred1))
                epoch_train_apr.append(average_precision_score(y_true, y_pred))
                epoch_best_thr.append(best_threshold)
                
            # Log training status after each batch
            logs["loss"] = epoch_train_loss.mean
            logs["auc"] = np.mean(epoch_train_auc)
            logs["prec"] = np.mean(epoch_train_prec)
            logs["rec"] = np.mean(epoch_train_rec)
            logs["apr"] = np.mean(epoch_train_apr)
            logs["thr"] = np.mean(epoch_best_thr)
            
            print(
                "Epoch {}, Train Batch {}, Loss {:.4f}, AUC {:.4f}, AUCPR {:.4f}, Precision {:.4f}, Recall {:.4f}".format(
                    epoch, bid, logs["loss"], logs["auc"], logs["apr"], logs["prec"], logs["rec"]
                )
            )
            global_steps += 1
        # Evaluate
        print("Start validation epoch:", epoch)
        model.eval()
        epoch_val_loss = Accumulator()
        epoch_val_prec = []
        epoch_val_rec = []
        epoch_val_auc = []
        epoch_val_apr = []
        
        for batch in valid_loader:
            batchsize = batch.x.shape[0]
            norm = T.NormalizeFeatures()
            batch = norm(batch).to(device)
            with torch.no_grad():
                # Forward pass
                batch.x = batch.x.type(torch.FloatTensor)
                batch.y = batch.y.type(torch.LongTensor)
                out = model(batch.x, batch.edge_index) 

                # Calculate loss
                class_weight = torch.FloatTensor([1.0, 20.0])
                valid_loss = F.cross_entropy(out[batch.is_validation], batch.y[batch.is_validation], class_weight)
                # f1_loss(batch.y[batch.is_validation], out[batch.is_validation])
                epoch_val_loss.update(valid_loss.item() * batchsize, batchsize)
                # Prediction
                pred = out.argmax(dim=1)
                y_pred = out[batch.is_validation][:,1].cpu().numpy()
                y_true = batch.y[batch.is_validation].cpu().numpy()
                # softmax = F.softmax(out, dim=1)[batch.is_validation][:,1].cpu().numpy()
                y_pred1 = (y_pred > np.mean(epoch_best_thr)).astype(int)
                
                try:
                    epoch_val_auc.append(roc_auc_score(y_true, y_pred))
                except:
                    epoch_val_auc.append(np.NaN)
                epoch_val_prec.append(precision_score(y_true, y_pred1))
                epoch_val_rec.append(recall_score(y_true, y_pred1))
                epoch_val_apr.append(average_precision_score(y_true, y_pred))

        # Log testing result after each epoch
        logs["val_loss"] = epoch_val_loss.mean
        logs["val_prec"] = np.mean(epoch_val_prec)
        logs["val_auc"] = np.mean(epoch_val_auc)
        logs["val_rec"] = np.mean(epoch_val_rec)
        logs["val_apr"] = np.mean(epoch_val_apr)
        print(
            "Epoch {}, Valid Loss {:.4f}, Valid AUC {:.4f}, Valid AUCPR {:.4f}, Valid Precision {:.4f}, Valid Recall {:.4f}".format(
                epoch, logs["val_loss"], logs["val_auc"], logs["val_apr"], logs["val_prec"], logs["val_rec"]
            )
        )

def train_pcgnn():
    return

if __name__ == "__main__":
    
    # get graph data
    G = get_graph()
    data = G.data


    train_baseline() #one class SVM or xgboost
    train_graphsage()
    train_node2vec()
    train_gcn()
    train_pcgnn()
