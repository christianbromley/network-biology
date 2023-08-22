#import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
#import PyWGCNA as wgcna
import sys

sys.path.insert(0, "./opt/anaconda3/lib/python3.8/site-packages")

import random
from scipy.linalg import expm


# def generate_coexpression_network(expr_mat: pd.DataFrame,
#                                   gene_mapping: pd.DataFrame,
#                                   sample_metadata: pd.DataFrame,
#                                   method: str = 'wgcna', **kwargs):
#     """
#     """
#     # create ann data object
#     adata = ad.AnnData(expr_mat)
#     adata.var = gene_mapping
#     adata.obs = sample_metadata
#
#     # create WGCNA object
#     wgcna_obj = wgcna.WGCNA(anndata=adata, **kwargs)
#     wgcna_obj.findModules()
#     wgcna_obj.saveWGCNA()
#
#     if method == 'wgcna':
#         network = networkx.Graph(wgcna_obj.adjacency)
#
#     return wgcna_obj, network


def weighted_coexpression_network(matrix: pd.DataFrame,
                                  directionality: bool = True,
                                  directionality_method: str = "positive_only",
                                  power: int = 1,
                                  cutoff: float = 0.5):
    '''
    Creates a co-expression network

    Args:
        matrix (pd.DataFrame): matrix of numeric values to co-correlate
        directionality (bool): defaults to True
        directionality_method (str): defaults to "positive_only" where only positive correlations used. 
        "positive_norm" mean values are scaled between 0 and 1. "positive_negative" means negative correlations are also encoded.
        
        power (int):
        cutoff (float):
    
    Returns:
        coexpression_network (networkx.Graph): networkx graph of the coexpression network

    '''
    # compute correlation coefficients
    correlation_matrix = matrix.corr()

    # a few different options:
    # if we want to encode directionality somehow
    # we can 
    # 1) just use positive correlations 
    # 2) normalise between 0 and 1
    # 3) use positive and negative values

    # if user wants to encode directionality
    if directionality:
        if directionality_method == "positive_only":
            # apply correlation coefficient cutoff
            correlation_matrix[correlation_matrix < cutoff] = 0
            adjacency_matrix = correlation_matrix.copy()
        else:
            # apply correlation coefficient cutoff
            correlation_matrix[(correlation_matrix < cutoff) & (correlation_matrix > (cutoff * -1))] = 0

            if directionality_method == "positive_norm":
                # normalise to be between 0 and 1 so that correlation of 0 is 0.5
                adjacency_matrix = (correlation_matrix + 1) / 2
            elif directionality_method == "positive_negative":
                adjacency_matrix = correlation_matrix.copy()
            else:
                raise ValueError("directionality_method is not one of positive_only, positive_norm or positive_negative")
    else:
        # convert correlation values to absolute values
        adjacency_matrix = np.abs(correlation_matrix)

        # Apply threshold to low correlation
        adjacency_matrix[adjacency_matrix < cutoff] = 0

    
    # apply a power, default is 1 such that no additional transformation is applied
    adjacency_matrix = adjacency_matrix ** power

    # remove self-correlation
    np.fill_diagonal(adjacency_matrix.values, 0)

    print(adjacency_matrix)

    # Generate mapping dict for gene symbols
    label_mapping = {i: symbol for i, symbol in enumerate(adjacency_matrix.columns)}

    # Convert to weighted graph
    coexpression_network = nx.from_numpy_array(adjacency_matrix.values)

    # Map labels back go gene names
    coexpression_network = nx.relabel_nodes(coexpression_network, label_mapping)

    # Remove nodes without edges
    coexpression_network.remove_nodes_from(list(nx.isolates(coexpression_network)))

    return coexpression_network


def get_edges_between_nodes(node_set, graph):
    """
    Returns a list of all the edges between a set of nodes in a NetworkX graph object.

    Args:
        node_set (set): the set of nodes to find edges between
        graph (networkx.Graph): the input graph object

    Returns:
        edges (list): a list of all the edges between the nodes in the node set
    """
    # Initialize an empty list to store the edges
    edges = []

    # Iterate over all the edges in the graph
    for u, v, d in graph.edges(data=True):
        # Check if both nodes are in the node set
        if u in node_set and v in node_set:
            # If so, add the edge to the list
            edges.append((u, v, d))

    # Return the list of edges
    return edges


def random_subgraph(graph, fraction):
    """
    Returns a random subset of nodes and edges from a NetworkX graph object, retaining all associated attributes.

    Args:
        graph (networkx.Graph): the input graph object
        fraction (float): the proportion of nodes and edges to retain (must be between 0 and 1)

    Returns:
        subgraph (networkx.Graph): the random subgraph of nodes and edges
    """
    # compute number of nodes for subgraph
    num_nodes = int(fraction * len(graph.nodes()))

    # randomly select num_node nodes
    nodes = np.random.choice(list(graph.nodes()), size=num_nodes, replace=False)

    # create a subgraph from the selected nodes
    subgraph = graph.subgraph(nodes)

    return subgraph


# In networkx:
# nx.clustering(coexpr_net, weight='weight')
# nx.average_clustering(coexpr_net, weight='weight')
# Change this function could be changed to just plot the distribution of the coefficients

def compute_weighted_clustering_coefficient(graph):
    """
    Computes the clustering coefficient for each node in a NetworkX weighted graph object, as well as the
    average clustering coefficient.

    Args:
        graph (networkx.Graph): the input graph object

    Returns:
        avg_cc (float): the average clustering coefficient of the graph
        cc_dict (dict): a dictionary containing the clustering coefficient for each node in the graph
    """
    # Compute the clustering coefficient for each node
    cc_dict = nx.clustering(graph, weight='weight')

    # get this in data frame format
    # cc_df = pd.DataFrame(cc_dict, index=cc_dict.keys()).unstack()
    cc_df = pd.DataFrame.from_dict(cc_dict, orient='index')
    cc_df.columns = ['clustering_coefficient']

    # Plot the degree distribution
    plt.hist(cc_df.clustering_coefficient.tolist(), bins=20)
    plt.xlabel('Clustering coefficient per node')
    plt.ylabel('Count')
    plt.title('Clustering coefficient')
    plt.show()

    # Compute the average clustering coefficient of the graph
    #nx.average_clustering()
    avg_cc = nx.average_clustering(graph, weight='weight')

    # Return the average clustering coefficient and the clustering coefficient dictionary
    return avg_cc, cc_df


def plot_degree_distribution(graph, seeds: str = None, **kwargs):
    """
    Plots the degree distribution for a NetworkX graph object.

    Args:
        graph (networkx.Graph): the input graph object
        seeds (list): list of node IDs or seeds
    
    Returns:
        node_degrees (pd.DataFrame): data frame with weighted degree of nodes
    """
    # Compute the degree of each node in the graph
    degrees = [graph.degree(node, weight='weight') for node in graph.nodes()]
    nodes = graph.nodes()
    node_degrees = pd.DataFrame({
        'node': nodes,
        'weighted_degree': degrees
    })

    # Plot the degree distribution
    plt.hist(node_degrees.weighted_degree.tolist(), bins=10, **kwargs)
    plt.xlabel('Degree (weighted)')
    plt.ylabel('Count')
    plt.title('Degree Distribution')
    plt.show()

    return node_degrees


#nx.modularity_matrix(graph, weight='weight')
def compute_modularity_matrix(graph):
    """
    Computes the modularity of a NetworkX weighted graph object.

    Args:
        graph (networkx.Graph): the input graph object

    Returns:
        modularity (float): the modularity of the graph
    """
    # Get the adjacency matrix and edge weights
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    edge_weights = np.array([d['weight'] for (u, v, d) in graph.edges(data=True)])

    # Compute the total weight of the graph
    total_weight = np.sum(edge_weights)

    # Compute the expected weight matrix
    expected_matrix = np.outer(np.sum(adj_matrix, axis=0), np.sum(adj_matrix, axis=1)) / (2 * total_weight)

    # Compute the modularity matrix
    mod_matrix = adj_matrix - expected_matrix

    # Return the modularity score
    return mod_matrix


def compute_centrality_metrics(graph):
    """
    Computes various centrality metrics for nodes in a NetworkX graph object.

    Parameters:
        graph (networkx.Graph): the input graph object

    Returns:
        centrality_df (pandas.DataFrame): a DataFrame containing the centrality metrics for each node
    """
    # Compute the degree centrality
    degree_centrality = nx.degree_centrality(graph)

    # Compute the eigenvector centrality
    eigenvector_centrality = nx.eigenvector_centrality(graph, weight='weight')

    # Compute the closeness centrality
    closeness_centrality = nx.closeness_centrality(graph, distance='weight')

    # Compute the betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(graph, weight='weight')

    # Create a DataFrame to store the results
    centrality_df = pd.DataFrame({
        'Node': list(graph.nodes),
        'Degree Centrality': list(degree_centrality.values()),
        'Eigenvector Centrality': list(eigenvector_centrality.values()),
        'Closeness Centrality': list(closeness_centrality.values()),
        'Betweenness Centrality': list(betweenness_centrality.values())
    })

    # Sort the DataFrame by node ID
    centrality_df.sort_values(by='Node', inplace=True)

    # Return the DataFrame
    return centrality_df


#https://netneurotools.readthedocs.io/en/latest/generated/netneurotools.metrics.communicability_wei.html
def compute_weighted_communicability(graph):
    """
    Computes the communicability of pairs of nodes in `adjacency`

    Args:
        graph (networkx graph): a graph with weighted edges

    Returns:
        cmc : (N, N) numpy.ndarray
            Symmetric array representing communicability of nodes {i, j}

    References:
        Crofts, J. J., & Higham, D. J. (2009). A weighted communicability measure
        applied to complex brain networks. Journal of the Royal Society Interface,
        6(33), 411-414.

    """
    # USE nx.to_numpy_array()
    adjacency = nx.to_numpy_array(graph)

    # negative square root of nodal degrees
    row_sum = adjacency.sum(1)
    neg_sqrt = np.power(row_sum, -0.5)
    square_sqrt = np.diag(neg_sqrt)

    # normalize input matrix
    for_expm = square_sqrt @ adjacency @ square_sqrt

    # calculate matrix exponential of normalized matrix
    cmc = expm(for_expm)
    cmc[np.diag_indices_from(cmc)] = 0

    return pd.DataFrame(cmc, index=graph.nodes, columns=graph.nodes)


def mean_weighted_communicability_between_nodesets(comms_matrix: pd.DataFrame, starting_nodes: list, ending_nodes: list):
    """
    Compute the mean communicability between two sets of nodes

    Args:
        comms_matrix (pd.DataFrame): matrix with weighted communicability matrix e.g. output from compute_weighted_communicability
        starting_nodes (list): names of the nodes to start at
        ending_nodes (list): names of the nodes to assess communication with

    """
    assert len(set(starting_nodes) & set(
        ending_nodes)) == 0, 'There must be no overlaps between your starting and ending nodes'

    # ensure we are using nodes that are actually in the communicability matrix
    starting_nodes = list(set(starting_nodes) & set(comms_matrix.index))
    ending_nodes = list(set(ending_nodes) & set(comms_matrix.index))

    mean_communication = pd.DataFrame({
        'starting_nodes': [starting_nodes],
        'ending_nodes': [ending_nodes],
        'starting': [np.mean(comms_matrix.loc[starting_nodes, starting_nodes].to_numpy())],
        'ending': [np.mean(comms_matrix.loc[ending_nodes, ending_nodes].to_numpy())],
        'starting_ending': [np.mean(comms_matrix.loc[starting_nodes, ending_nodes].to_numpy())]
    })

    return mean_communication


def ablate_and_communicate(graph, **kwargs):
    """
    Ablate each node and then assess communicability between two sets of nodes

    Parameters:
        graph: networkx object

    Returns:
        data frame with communication following ablation
    """
    # first without ablation
    comms = compute_weighted_communicability(graph)
    mean_communication = compute_mean_weighted_communicability(comms, **kwargs)
    mean_communication['iteration'] = 'full_graph'

    # iterate through nodes ablating all of them
    nodes = graph.nodes()
    for node in nodes:
        G = graph.copy()
        G.remove_node(node)

        comms_sub = compute_weighted_communicability(G)
        mean_communication_sub = compute_mean_weighted_communicability(comms_sub, **kwargs)
        mean_communication_sub['iteration'] = node

        mean_communication = pd.concat([mean_communication, mean_communication_sub], axis=0)

    return mean_communication
