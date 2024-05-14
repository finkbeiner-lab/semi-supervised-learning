import numpy as np
from scipy.sparse import csr_matrix
import faiss
import igraph as ig
import leidenalg
from multiprocessing import Pool
import matplotlib.pyplot as plt

def leiden_clustering_sparse(matrix_chunk):
    """
    Perform Leiden clustering on a given data matrix chunk.
    
    Parameters
    ----------
    matrix_chunk : tuple
        Tuple containing a chunk of the input data matrix, where each row is a sample and each column is a feature.

    Returns
    -------
    labels : np.ndarray
        Cluster labels assigned to each sample in the chunk.
    """
    matrix, n_neighbors, resolution = matrix_chunk
    # Step 1: Compute nearest neighbors using FAISS and Euclidean distance
    print('Finding Neighbors')
    d = matrix.shape[1]  # dimension
    index = faiss.IndexFlatIP(d)  # L2 (Euclidean) index
    index.add(matrix)  # Add dataset to index

    _, neighbors = index.search(matrix, n_neighbors + 1)  # +1 because the query itself is always returned

    # Create sparse adjacency matrix
    n_samples = matrix.shape[0]
    row_indices = np.repeat(np.arange(n_samples), n_neighbors)
    col_indices = neighbors[:, 1:].flatten()  # Exclude the first column (self-connections)
    print("Row indices:", row_indices)
    print("Col indices:", col_indices)
    data = np.ones(n_neighbors * n_samples)
    adj_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_samples))

    # Step 2: Create a graph from the sparse adjacency matrix
    print('Constructing graph')
    edge_list = list(zip(adj_matrix.nonzero()[0], adj_matrix.nonzero()[1]))
    G = ig.Graph(edges=edge_list, directed=False)

    # Step 3: Perform Leiden clustering
    print('Partitioning graph')
    partition = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition,
                                         resolution_parameter=resolution)

    labels = np.array(partition.membership)
    return labels

# Load embeddings from file
embeddings = np.load("embeddings.npy")

# Parameters for clustering
n_neighbors = 5
resolution = 0.5
num_processes = 34  # Adjust according to the number of CPU cores available

# Split embeddings into chunks for parallel processing
chunk_size = len(embeddings) // num_processes
embedding_chunks = [(embeddings[i:i+chunk_size], n_neighbors, resolution) for i in range(0, len(embeddings), chunk_size)]

# Initialize Pool for parallel processing

with Pool(processes=num_processes) as pool:
    cluster_labels_chunks = pool.map(leiden_clustering_sparse, embedding_chunks)

# Concatenate cluster labels from all chunks
cluster_labels = np.concatenate(cluster_labels_chunks)

# Plot the cluster labels
plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.title('Leiden Clustering')
plt.xlabel('Embedding Dimension 1')
plt.ylabel('Embedding Dimension 2')
plt.colorbar(label='Cluster Label')
plt.savefig('leiden.jpg')
plt.show()
