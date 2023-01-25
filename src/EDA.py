# Exploring the Ethereum transaction graph with GSQL queries
import pyTigerGraph as tg
import json

def get_counts(connection):
    """
    Helper function to get vertex and edge counts for the connected graph.
    """
    print("Vertex Counts")
    for vertex in connection.getVertexTypes():
        print(f"There are {connection.getVertexCount(vertex)} {vertex} vertices in the graph")

    print("--------------")
    print("Edge Counts")
    for edge in connection.getEdgeTypes():
        print(f"There are {connection.getEdgeCount(edge)} {edge} edges in the graph")

def install_queries(connection):
    """
    Helper function to install user queries to the connected TigerGraph instance.
    """
    connection.gsql('''
        USE GRAPH Ethereum
        INSTALL QUERY hashtags_from_person
    ''')

def run_queries(connection):
    """
    Helper function to run user queries on the connected TigerGraph instance.
    """
    results = connection.runInstalledQuery("hashtags_from_person", params={"inPer": "50"})
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    get_counts(conn)
    install_queries(conn)
    run_queries(conn)
    