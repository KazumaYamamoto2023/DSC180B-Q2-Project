# connect to TigerGraph database instance
import pyTigerGraph as tg
import json
import os

# import TigerGraph instance config
os.chdir('../config/')
with open('tigergraph.json', 'r') as f:
    config = json.load(f)

def connect_db(graph_name):
    # Connection parameters to an initial TigerGraph GraphStudio Graph
    conn = tg.TigerGraphConnection(host=config['host'], gsqlSecret=config['secret'], graphname=graph_name)
    conn.getToken(config['secret'])
    return conn

def publish_schema(connection):
    # Publish Global View Graph Schema
    connection.gsql('''
        USE GLOBAL
        CREATE VERTEX Wallet (PRIMARY_ID id INT, label FLOAT) WITH primary_id_as_attribute="true"
        CREATE DIRECTED EDGE sent_eth (from Wallet, to Wallet, amount FLOAT, sent_date INT) WITH REVERSE_EDGE="reverse_sent_eth"
        CREATE DIRECTED EDGE received_eth (from Wallet, to Wallet, amount FLOAT, receive_date INT) WITH REVERSE_EDGE="reverse_received_eth"
    ''')

def build_graph(connection):
    # Create a new graph from the global schema
    connection.gsql('''
        CREATE GRAPH MyGraph2(Wallet, sent_eth, reverse_sent_eth, received_eth, reverse_received_eth)
    ''')

def connect_graph(connection, graph_name):
    # Connect to a newly created graph
    connection.graphname=graph_name"
    secret = connection.createSecret()
    connection = tg.TigerGraphConnection(host=config['host'], gsqlSecret=config['secret'], graphname=graph_name)
    connection.getToken(secret)
    connection.getSchema()
    return connection

def load_data(connection, nodes_file, edges_file):
    # Custom loading job that maps the values of nodes.csv to VERTEX attributes
    connection.gsql('''
        USE GRAPH MyGraph2
        BEGIN
        CREATE LOADING JOB load_wallets FOR GRAPH MyGraph2 {
        DEFINE FILENAME MyDataSource;
        LOAD MyDataSource TO VERTEX Wallet VALUES($0, $1) USING SEPARATOR=",", HEADER="true", EOL="\\n", QUOTE="double";
        }
        END
    ''')

    # Custom loading job that maps the values of edges.csv to EDGE attributes
    connection.gsql('''
        USE GRAPH MyGraph2
        BEGIN
        CREATE LOADING JOB load_transactions FOR GRAPH MyGraph2 {
        DEFINE FILENAME MyDataSource;
        LOAD MyDataSource TO EDGE sent_eth VALUES($1, $0, $2, $3) USING SEPARATOR=",", HEADER="true", EOL="\\n";
        LOAD MyDataSource TO EDGE received_eth VALUES($0, $1, $2, $3) USING SEPARATOR=",", HEADER="true", EOL="\\n";
        }
        END
    ''')

    connection.runLoadingJobWithFile(filePath=nodes_file, fileTag='MyDataSource', jobName='load_wallets')
    connection.runLoadingJobWithFile(filePath=edges_file, fileTag='MyDataSource', jobName='load_transactions')

def main():
    # Nodes/edges files:
    nodes_file = 'data/nodes.csv'
    edges_file = 'data/edges.csv'

    # Build and upload graph to TigerGraph
    conn = connect_db("MyGraph")
    publish_schema(conn)
    build_graph(conn)
    conn_graph = connect_graph(conn, "MyGraph2")
    load_data(conn_graph, nodes_file, edges_file)
    return

if __name__ == "__main__":
    main()