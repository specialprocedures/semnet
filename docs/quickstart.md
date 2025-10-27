# Quick Start

## Basic Usage

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
sem = SemanticNetwork(thresh=0.3, verbose=True)  # Larger values give sparser networks

# Build the semantic graph from your embeddings
G = sem.fit_transform(embeddings, labels=docs)

# Export to pandas
nodes_df, edges_df = to_pandas(G)

# Analyze the graph
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Connected components: {nx.number_connected_components(G)}")

# Find similar document groups
for component in nx.connected_components(G):
    if len(component) > 1:
        similar_docs = [G.nodes[i]["label"] for i in component]
        print(f"Similar documents: {similar_docs}")

# Calculate centrality measures,
# Degree centrality not that interesting in the example, but shown here for demonstration
centrality = nx.degree_centrality(G)
for node, cent_value in centrality.items():
    print(f"Document: {G.nodes[node]['label']}, Degree Centrality: {cent_value:.4f}")
    G.nodes[node]["degree_centrality"] = cent_value


```