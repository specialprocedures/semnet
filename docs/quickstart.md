# Quick Start

## Installation

You can easily install Semnet with `pip`.

```bash
pip install semnet
```

## Basic Usage

All you need to start building your network is a set of embeddings and (optionally) some labels.

```python
from semnet import SemanticNetwork, to_pandas
from sentence_transformers import SentenceTransformer
import networkx as nx

# Your documents
docs = [
    "The cat sat on the mat",
    "A cat was sitting on a mat",
    "The dog ran in the park",
    "I love Python",
    "Python is a great programming language",
]

# Generate embeddings (use any embedding provider)
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = embedding_model.encode(docs)

# Create and configure semantic network
sem = SemanticNetwork(thresh=0.3, distance="angular")

# Build the semantic graph from your embeddings
G = sem.fit_transform(embeddings, labels=docs)

# Analyze the graph
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Connected components: {nx.number_connected_components(G)}")

# Export to pandas
nodes_df, edges_df = to_pandas(G)

# Find similar document groups
for component in nx.connected_components(G):
    if len(component) > 1:
        similar_docs = [G.nodes[i]["label"] for i in component]
        print(f"Similar documents: {similar_docs}")

# Calculate centrality measures
centrality = nx.degree_centrality(G)
for node, cent_value in centrality.items():
    print(f"Document: {G.nodes[node]['label']}, Degree Centrality: {cent_value:.4f}")
    G.nodes[node]["degree_centrality"] = cent_value
```

## Performance at Scale

### The Scaling Problem

Graph construction entails finding pairwise relationships (edges) between entities (nodes) in a dataset. For large corpora, scaling problems rapidly become apparent as the number of possible pairs in a set scales quadratically.

If we were to naively attempt to construct a graph from a modestly-sized set of documents we encounter problems early on. Building a graph from 10,000 documents would entail operations across 50 million pairs, for 100,000 it's around 5 billion!

### Naive Approach Problems

```python
from sklearn.metrics import DistanceMetric
import numpy as np

dist = DistanceMetric.get_metric("euclidean")

# Generate 100,000 random embeddings
embeddings = np.random.rand(100_000, 768)
dist_scores = dist.pairwise(embeddings)

>> MemoryError: Unable to allocate 74.5 GiB for an array
with shape (100000, 100000) and data type float64
```

### Semnet's Solution

Semnet solves the scaling problem using [Approximate Nearest Neighbours](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor) search with [Annoy](https://github.com/spotify/annoy).

Instead of making comparisons between each document in the corpus, Semnet indexes the embeddings, iterates over each one, and returns a `top_k` best matches from within their neighbourhood.

```python
from semnet import SemanticNetwork
import time

start_time = time.time()

# Make 100,000 embeddings
embeddings = np.random.rand(100_000, 768)

# Build semantic network
semnet = SemanticNetwork(thresh=0.4, top_k=5)
G = semnet.fit_transform(embeddings)

end_time = time.time()
print(f"Processing time: {end_time - start_time:.2f} seconds")

>> Processing time: 24.26 seconds
```

We're not only able to process all the embeddings without crashing our kernel, but it's done in under 30 seconds.

## Working with Node Data

You can pass arbitrary metadata during graph construction:

```python
# Create node data dictionary
node_data = {}
for i, doc in enumerate(docs):
    node_data[i] = {
        "category": "python" if "python" in doc.lower() else "other",
        "length": len(doc),
        "custom_field": "example"
    }

# Build graph with metadata
G = sem.fit_transform(
    embeddings=embeddings,
    labels=docs,
    node_data=node_data
)

# Access node attributes
for node_id, node_attrs in G.nodes(data=True):
    print(f"Node {node_id}: {node_attrs}")
```

## Graph Analysis

Once you have your graph, you can leverage the full power of NetworkX:

```python
import networkx as nx

# Community detection
communities = nx.community.louvain_communities(G)
print(f"Found {len(communities)} communities")

# Centrality measures
degree_centrality = nx.degree_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Path finding
if G.number_of_nodes() > 2:
    nodes = list(G.nodes())
    try:
        path = nx.shortest_path(G, nodes[0], nodes[1])
        print(f"Shortest path: {path}")
    except nx.NetworkXNoPath:
        print("No path found between nodes")
```