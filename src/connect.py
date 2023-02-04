# connect to TigerGraph database instance
import pyTigerGraph as tg
import json

def build_schema():
    """
    Prior to running this function, please create a graph called 'Ethereum' in TigerGraph GraphStudio,
    and create a gsqlSecret through TigerGraph's AdminPortal. This function will connect to TigerGraph instance,
    publish the graph schema, creates a graph with the schema, and load wallet and transaction data into the graph.
    """
    # Import TigerGraph instance configuration
    with open('config/tigergraph.json', 'r') as f:
        config = json.load(f)

    # Connect to TigerGraph instance
    hostName = config['host']
    secret = config['secret']
    conn = tg.TigerGraphConnection(host=hostName, gsqlSecret=secret, graphname="Ethereum")
    conn.getToken(secret)

    # Publish graph schema to TigerGraph
    print(conn.gsql(open("gsql/build_schema.gsql", "r").read()))

    # Create data loading jobs
    print(conn.gsql(open("gsql/load_data.gsql", "r").read()))

    # Upload graph data to TigerGraph
    print(conn.runLoadingJobWithFile('data/nodes.csv', "MyDataSource", "load_wallets"))
    print(conn.runLoadingJobWithFile('data/edges.csv', "MyDataSource", "load_transactions"))

    # Print graph schema
    print("Vertex Counts")
    for vertex in conn.getVertexTypes():
        print(f"There are {conn.getVertexCount(vertex)} {vertex} vertices in the graph")

    print("--------------")
    print("Edge Counts")
    for edge in conn.getEdgeTypes():
        print(f"There are {conn.getEdgeCount(edge)} {edge} edges in the graph")
    return

def add_features():
    """
    Prior to running this function, please create a graph called 'Ethereum' in TigerGraph GraphStudio,
    and create a gsqlSecret through TigerGraph's AdminPortal. This function will connect to TigerGraph instance,
    install the GSQL queries to the database, run the queries on the 'Ethereum' transaction graph, and save the query
    results as node attributes.
    """
    # Import TigerGraph instance configuration
    with open('config/tigergraph.json', 'r') as f:
        config = json.load(f)

    # Connect to TigerGraph instance
    hostName = config['host']
    secret = config['secret']
    conn = tg.TigerGraphConnection(host=hostName, gsqlSecret=secret, graphname="Ethereum")
    conn.getToken(secret)

    # Install and run GSQL query for indegree/outdegree
    f = conn.gds.featurizer()
    f.installAlgorithm("get_degrees", query_path="gsql/get_degrees.gsql")
    print(json.dumps(f.runAlgorithm("get_degrees", custom_query=True), indent=2))

    # Install and run GSQL query for total/minimum ETH sent/received
    f = conn.gds.featurizer()
    f.installAlgorithm("get_total_amount", query_path="gsql/get_total_amount.gsql")
    print(json.dumps(f.runAlgorithm("get_total_amount", custom_query=True), indent=2))

    # Install and run GSQL query for pagerank
    f = conn.gds.featurizer()
    f.installAlgorithm("tg_pagerank")
    tg_pagerank_params = {
        "v_type": "Wallet",
        "e_type": "sent_eth",
        "result_attribute": "pagerank",
        "top_k":5  
    }
    print(json.dumps(f.runAlgorithm("tg_pagerank",tg_pagerank_params)[0]['@@top_scores_heap'], indent=2))
    return