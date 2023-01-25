# connect to TigerGraph database instance
import pyTigerGraph as tg
import json
import os

def connect_db(host, secret, graph_name):
    """
    Helper function to intialize connection to a TigerGraph GraphStudio graph.
    """
    # Connection parameters to an initial TigerGraph GraphStudio Graph
    conn = tg.TigerGraphConnection(host=host, gsqlSecret=secret, graphname=graph_name)
    conn.getToken(secret)
    return conn

def publish_schema(connection):
    """
    Helper function to publish the global graph schema to TigerGraph GraphStudio.
    """
    # Publish Global View Graph Schema
    connection.gsql('''
        USE GLOBAL
        CREATE VERTEX Wallet (PRIMARY_ID id INT, label FLOAT) WITH primary_id_as_attribute="true"
        CREATE DIRECTED EDGE sent_eth (from Wallet, to Wallet, amount FLOAT, sent_date INT) WITH REVERSE_EDGE="reverse_sent_eth"
        CREATE DIRECTED EDGE received_eth (from Wallet, to Wallet, amount FLOAT, receive_date INT) WITH REVERSE_EDGE="reverse_received_eth"
    ''')

def build_graph(connection):
    """
    Helper function to create a new graph in TigerGraph using the global graph schema.
    """
    # Create a new graph from the global schema
    connection.gsql('''
        CREATE GRAPH Ethereum(Wallet, sent_eth, reverse_sent_eth, received_eth, reverse_received_eth)
    ''')

def connect_graph(connection, host, secret, graph_name):
    """
    Helper function to connect to a newly created graph in TigerGraph GraphStudio.
    """
    # Connect to a newly created graph
    connection.graphname=graph_name
    new_secret = connection.createSecret()
    connection = tg.TigerGraphConnection(host=host, gsqlSecret=secret, graphname=graph_name)
    connection.getToken(new_secret)
    connection.getSchema()
    return connection

def load_data(connection, nodes_file, edges_file):
    """
    Helper function that defines custom loading jobs to map values from nodes/edges files
    to vertex/edge attributes in our transaction graph.
    """
    # Custom loading job that maps the values of nodes.csv to VERTEX attributes
    connection.gsql('''
        USE GRAPH Ethereum
        BEGIN
        CREATE LOADING JOB load_wallets FOR GRAPH Ethereum {
        DEFINE FILENAME MyDataSource;
        LOAD MyDataSource TO VERTEX Wallet VALUES($0, $1) USING SEPARATOR=",", HEADER="true", EOL="\\n", QUOTE="double";
        }
        END
    ''')

    # Custom loading job that maps the values of edges.csv to EDGE attributes
    connection.gsql('''
        USE GRAPH Ethereum
        BEGIN
        CREATE LOADING JOB load_transactions FOR GRAPH Ethereum {
        DEFINE FILENAME MyDataSource;
        LOAD MyDataSource TO EDGE sent_eth VALUES($1, $0, $2, $3) USING SEPARATOR=",", HEADER="true", EOL="\\n";
        LOAD MyDataSource TO EDGE received_eth VALUES($0, $1, $2, $3) USING SEPARATOR=",", HEADER="true", EOL="\\n";
        }
        END
    ''')

    connection.runLoadingJobWithFile(filePath=nodes_file, fileTag='MyDataSource', jobName='load_wallets')
    connection.runLoadingJobWithFile(filePath=edges_file, fileTag='MyDataSource', jobName='load_transactions')

def main():
    """
    Connect to TigerGraph instance, publish the global graph schema, create a new graph from the global schema,
    and load wallet and transaction data into the graph.
    """
    # Import TigerGraph instance configuration
    os.chdir('../config/')
    with open('tigergraph.json', 'r') as f:
        config = json.load(f)
    
    # Nodes/edges files:
    os.chdir('../data/')
    nodes_file = 'nodes.csv'
    edges_file = 'edges.csv'

    # Publish global schema, create new graph, and upload data to TigerGraph
    conn = connect_db("MyGraph", config['host'], config['secret'])
    publish_schema(conn)
    build_graph(conn)
    conn_graph = connect_graph(conn, "Ethereum", config['host'], config['secret'])
    load_data(conn_graph, nodes_file, edges_file)
    return

if __name__ == "__main__":
    main()