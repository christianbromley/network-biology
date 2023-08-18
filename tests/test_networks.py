import numpy as np
import os
import pandas as pd
import sys

#print(sys.path)
#print(os.getcwd())
sys.path.insert(0, f"{os.getcwd()}/src")
#print(sys.path)

from networks import weighted_coexpression_network

# create a dummy dataset
np.random.seed(44)

# Number of features and observations
num_features = 10
num_observations = 10

# Generate random data for the matrix
mock_data = np.random.rand(num_observations, num_features)

# now generate a few different networks
mock_data = pd.DataFrame(mock_data)

# generate network 1 - directionalioty
net1 = weighted_coexpression_network(mock_data, directionality=True, directionality_method="positive_only", power=1, cutoff=0.2)

print(f"Number of nodes in network 1: {len(net1.nodes)}")
print(f"Number of edges in network 1: {len(net1.edges)}")

# generate network 2 - no directionality
net2 = weighted_coexpression_network(mock_data, directionality=True, directionality_method="positive_norm", power=1, cutoff=0.2)

print(f"Number of nodes in network 2: {len(net2.nodes)}")
print(f"Number of edges in network 2: {len(net2.edges)}")

# generate network 3 - no directionality
net3 = weighted_coexpression_network(mock_data, directionality=True, directionality_method="positive_negative", power=1, cutoff=0.2)

print(f"Number of nodes in network 3: {len(net3.nodes)}")
print(f"Number of edges in network 3: {len(net3.edges)}")


# generate network 3 - no directionality
net4 = weighted_coexpression_network(mock_data, directionality=False, directionality_method=None, power=1, cutoff=0.2)

print(f"Number of nodes in network 4: {len(net4.nodes)}")
print(f"Number of edges in network 4: {len(net4.edges)}")
