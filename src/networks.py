import anndata as ad
import matplotlib.pyplot as plt
import networkx
import numpy as np
import pandas as pd
import PyWGCNA as wgcna
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


def generate_coexpression_network(matrix: pd.DataFrame,
                                  directionality: bool = True,
                                  power: int = 1,
                                  cutoff: float = 0.5):
    '''
    Creates a co-expression network

    Args:
        matrix (pd.DataFrame):
        directionality (bool):
        power (int):
        cutoff (float):
    '''
    # compute correlation coefficients
    correlation_matrix = matrix.corr()

    # if user wants to encode directionality
    if directionality:
        # apply correlation cutoff
        correlation_matrix[(correlation_matrix < cutoff) & (correlation_matrix > (cutoff * -1))] = 0

        # normalise to be between 0 and 1 so that correlation of 0 is 0.5
        adjacency_matrix = (correlation_matrix + 1) / 2

    else:
        # convert correlation values to absolute values
        adjacency_matrix = np.abs(correlation_matrix)

        # Apply threshold to low correlation
        adjacency_matrix[adjacency_matrix < cutoff] = 0

    
    # apply a power, default is 1 such that no additional transformation is applied
    adjacency_matrix = adjacency_matrix ** power

    # remove self-correlation
    np.fill_diagonal(adjacency_matrix.values, 0)

    # Generate mapping dict for gene symbols
    label_mapping = {i: symbol for i, symbol in enumerate(adjacency_matrix.columns)}

    # Convert to weighted graph
    coexpression_network = nx.from_numpy_matrix(adjacency_matrix.values)

    # Map labels back go gene names
    coexpression_network = nx.relabel_nodes(coexpression_network, label_mapping)

    # Remove nodes without edges
    coexpression_network.remove_nodes_from(list(nx.isolates(coexpression_network)))

    return coexpression_network


def get_edges_between_nodes(node_set, graph):
    """
    Returns a list of all the edges between a set of nodes in a NetworkX graph object.

    Parameters:
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


# def random_subset(graph, p):
#     """
#     Returns a random subset of nodes and edges from a NetworkX graph object, while retaining the edge weights.
#
#     Parameters:
#         graph (networkx.Graph): the input graph object
#         p (float): the proportion of nodes and edges to retain (must be between 0 and 1)
#
#     Returns:
#         random_graph (networkx.Graph): the random subset of nodes and edges
#     """
#     # Get a list of all the nodes in the graph
#     nodes = list(graph.nodes())
#
#     # Compute the number of nodes to retain
#     n_nodes = int(len(nodes) * p)
#
#     # Select a random subset of nodes to retain
#     random_nodes = random.sample(nodes, n_nodes)
#
#     edges = get_edges_between_nodes(set(random_nodes), graph=graph)
#
#     # Create a new graph object containing only the selected nodes and edges
#     random_graph = networkx.Graph()
#     random_graph.add_nodes_from(random_nodes)
#     random_graph.add_edges_from(edges)
#
#     # Return the random subset of nodes and edges
#     return random_graph


def random_subgraph(graph, fraction):
    """
    Returns a random subset of nodes and edges from a NetworkX graph object, retaining all associated attributes.

    Parameters:
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

    Parameters:
        graph (networkx.Graph): the input graph object

    Returns:
        avg_cc (float): the average clustering coefficient of the graph
        cc_dict (dict): a dictionary containing the clustering coefficient for each node in the graph
    """
    # Compute the clustering coefficient for each node
    cc_dict = networkx.clustering(graph, weight='weight')

    # get this in data frame format
    # cc_df = pd.DataFrame(cc_dict, index=cc_dict.keys()).unstack()
    cc_df = pd.DataFrame.from_dict(cc_dict, orient='index')
    cc_df.columns = ['clustering_coefficient']

    # Plot the degree distribution
    plt.hist(cc_df.clustering_coefficient.tolist(), bins=20)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Clustering coefficient per node')
    plt.ylabel('Count')
    plt.title('Clustering coefficient')
    plt.show()

    # Compute the average clustering coefficient of the graph
    #nx.average_clustering()
    avg_cc = sum(cc_dict.values()) / len(cc_dict)

    # Return the average clustering coefficient and the clustering coefficient dictionary
    return avg_cc, cc_df


def plot_degree_distribution(graph, seeds: str = None):
    """
    Plots the degree distribution for a NetworkX graph object.

    Parameters:
        graph (networkx.Graph): the input graph object
        seeds (list): list of ensembl IDs as seeds
    """
    # Compute the degree of each node in the graph
    degrees = [graph.degree(node, weight='weight') for node in graph.nodes()]
    nodes = graph.nodes()
    node_degrees = pd.DataFrame({
        'node': nodes,
        'weighted_degree': degrees
    })
    if seeds is not None:
        node_degrees['geneset'] = np.where(node_degrees['node'].isin(seeds), 'seeds', 'other')

    # Plot the degree distribution
    plt.hist(node_degrees.weighted_degree.tolist(), bins=10)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Degree (weighted)')
    plt.ylabel('Count')
    plt.title('Degree Distribution')
    plt.show()

    return node_degrees


#nx.modularity_matrix(graph, weight='weight')
def compute_modularity_matrix(graph):
    """
    Computes the modularity of a NetworkX weighted graph object.

    Parameters:
        graph (networkx.Graph): the input graph object

    Returns:
        modularity (float): the modularity of the graph
    """
    # Get the adjacency matrix and edge weights
    adj_matrix = networkx.adjacency_matrix(graph).toarray()
    edge_weights = np.array([d['weight'] for (u, v, d) in graph.edges(data=True)])

    # Compute the total weight of the graph
    total_weight = np.sum(edge_weights)

    # Compute the expected weight matrix
    expected_matrix = np.outer(np.sum(adj_matrix, axis=0), np.sum(adj_matrix, axis=1)) / (2 * total_weight)
    #expected_matrix = np.outer(np.sum(adj_matrix, axis=0), np.sum(adj_matrix, axis=1)) / total_weight

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
    degree_centrality = networkx.degree_centrality(graph)

    # Compute the eigenvector centrality
    eigenvector_centrality = networkx.eigenvector_centrality(graph, weight='weight')

    # Compute the closeness centrality
    closeness_centrality = networkx.closeness_centrality(graph, distance='weight')

    # Compute the betweenness centrality
    betweenness_centrality = networkx.betweenness_centrality(graph, weight='weight')

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

    Parameters
    ----------
    graph (networkx graph): a graph with weighted edges

    Returns
    -------
    cmc : (N, N) numpy.ndarray
        Symmetric array representing communicability of nodes {i, j}

    References
    ----------
    Crofts, J. J., & Higham, D. J. (2009). A weighted communicability measure
    applied to complex brain networks. Journal of the Royal Society Interface,
    6(33), 411-414.

    """
    # USE nx.to_numpy_array()
    adjacency = networkx.adjacency_matrix(graph).toarray()

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


def compute_mean_weighted_communicability(comms_matrix, starting_nodes, ending_nodes):
    """

    """
    assert len(set(starting_nodes) & set(
        ending_nodes)) == 0, 'There must be no overlaps between your starting and ending nodes'

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
    Ablate each node and then assess communicability

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
