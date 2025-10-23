````markdown
# Semnet: Semantic Networks from Embeddings

Semnet constructs semantic network graphs from embeddings, enabling graph-based analysis and operations over embedded documents, images, and more.

Semnet uses [annoy](https://github.com/spotify/annoy) to perform rapid pair-wise distance calculations across all embeddings in the dataset, then constructs NetworkX graphs where edges represent semantic similarity relationships.

## Overview

Semnet provides semantic graph construction using three key components:

1. **User-Provided Embeddings** - Bring your own embeddings from any source (OpenAI, sentence-transformers, etc.)
2. **Approximate Nearest Neighbor Search** - Efficiently finds similar documents using Annoy indexing
3. **Graph Construction** - Builds similarity graphs with configurable thresholds for network analysis

This design gives you complete flexibility over the embedding generation process while leveraging Semnet's optimized similarity search and graph construction.

## Quick Start

```python
from semnet import SemanticNetwork
from sentence_transformers import SentenceTransformer  # or any embedding provider
import networkx as nx

# Your documents
docs = [
    "The cat sat on the mat",
    "A cat was sitting on a mat",  # Similar to first
    "The dog ran in the park", 
    "Python is great for ML",
    "Machine learning with Python"  # Similar to fourth
]

# Generate embeddings (use any embedding method you prefer)
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = embedding_model.encode(docs)

# Create and configure semantic network
network = SemanticNetwork(
    thresh=0.8,
    verbose=True
)

# Build the semantic graph from your embeddings
graph = network.fit_transform(embeddings, labels=docs)

# Analyze the graph
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
print(f"Connected components: {nx.number_connected_components(graph)}")

# Find similar document groups
for component in nx.connected_components(graph):
    if len(component) > 1:
        similar_docs = [graph.nodes[i]['name'] for i in component]
        print(f"Similar documents: {similar_docs}")

# Export for further analysis
nodes_df, edges_df = network.to_pandas(graph)
```

## Features

- **Scikit-learn style API** - Familiar fit/transform interface for ML practitioners
- **Bring your own embeddings** - Use any embedding source (OpenAI, Cohere, sentence-transformers, etc.)
- **Configurable similarity thresholds** - Control graph density and connection strength
- **Fast similarity search** - Uses Annoy for efficient approximate nearest neighbor search
- **NetworkX integration** - Full compatibility with NetworkX graph algorithms and analysis
- **Verbose mode** - Progress bars and detailed logging
- **Pandas export** - Easy integration with data analysis workflows

## Installation

```bash
pip install semnet
```

For development:

```bash
git clone https://github.com/specialprocedures/semnet.git
cd semnet
pip install -e ".[dev]"
```

## Step-by-Step Usage

### 1. Basic Graph Construction

```python
from semnet import SemanticNetwork
from sentence_transformers import SentenceTransformer

# Generate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(your_documents)

# Create network and build graph
network = SemanticNetwork(thresh=0.8)
graph = network.fit_transform(embeddings, labels=your_documents)
```

### 2. Separate Fit and Transform

```python
from sentence_transformers import SentenceTransformer

# Generate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs)

network = SemanticNetwork()

# Fit the model on embeddings
network.fit(embeddings, labels=docs)

# Transform with different thresholds
strict_graph = network.transform(thresh=0.9)  # Fewer, stronger connections
loose_graph = network.transform(thresh=0.5)   # More, weaker connections
```

### 3. Using Different Embedding Sources

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Option 1: Use sentence-transformers
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs)

# Option 2: Use OpenAI embeddings (requires openai package)
# import openai
# embeddings = openai.Embedding.create(input=docs, model="text-embedding-ada-002")

# Option 3: Use your own pre-computed embeddings
# custom_embeddings = np.random.rand(len(docs), 384)  # Shape: (n_docs, embedding_dim)

network = SemanticNetwork(thresh=0.8)
graph = network.fit_transform(embeddings, labels=docs)
```

### 4. Advanced Node Data and IDs

```python
from sentence_transformers import SentenceTransformer

docs = ["Document 1", "Document 2", "Document 3"]
custom_ids = ["doc_001", "doc_002", "doc_003"]

# Additional metadata for each document
node_data = {
    0: {"category": "tech", "priority": 1, "author": "Alice"},
    1: {"category": "science", "priority": 2, "author": "Bob"}, 
    2: {"category": "tech", "priority": 1, "author": "Charlie"}
}

# Generate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs)

network = SemanticNetwork(thresh=0.8)
graph = network.fit_transform(
    embeddings, 
    labels=docs, 
    ids=custom_ids, 
    node_data=node_data
)

# Access additional data through the graph
for node_id in graph.nodes():
    node = graph.nodes[node_id]
    print(f"ID: {node['id']}, Category: {node['category']}, Author: {node['author']}")
```

### 5. Graph Analysis

```python
from sentence_transformers import SentenceTransformer
import networkx as nx

docs = ["Document 1", "Document 2", "Document 3"]
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs)

network = SemanticNetwork(thresh=0.8)
graph = network.fit_transform(embeddings, labels=docs)

# Basic graph metrics
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
print(f"Density: {nx.density(graph):.3f}")
print(f"Connected components: {nx.number_connected_components(graph)}")

# Find connected components (similar document groups)
for component in nx.connected_components(graph):
    if len(component) > 1:
        docs_in_component = [graph.nodes[i]['name'] for i in component]
        print(f"Similar documents: {docs_in_component}")

# Node centrality measures
centrality = nx.degree_centrality(graph)
most_central = max(centrality, key=centrality.get)
print(f"Most central document: {graph.nodes[most_central]['name']}")

# Clustering coefficient
clustering = nx.clustering(graph)
avg_clustering = sum(clustering.values()) / len(clustering)
print(f"Average clustering coefficient: {avg_clustering:.3f}")
```

### 6. Exporting to Pandas

```python
from sentence_transformers import SentenceTransformer

docs = ["Document 1", "Document 2", "Document 3"]
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs)

# Add some metadata
node_data = {
    0: {"author": "Alice", "category": "tech"},
    1: {"author": "Bob", "category": "science"},
    2: {"author": "Charlie", "category": "tech"}
}

network = SemanticNetwork(thresh=0.8)
graph = network.fit_transform(embeddings, labels=docs, node_data=node_data)

# Export main graph to pandas DataFrames
nodes_df, edges_df = network.to_pandas(graph)

# Export a subgraph (useful for focused analysis)
largest_component = max(nx.connected_components(graph), key=len)
subgraph = graph.subgraph(largest_component)
sub_nodes_df, sub_edges_df = network.to_pandas(subgraph)

# Analyze nodes
print("Nodes DataFrame:")
print(nodes_df.head())
print(f"Columns: {list(nodes_df.columns)}")

# Analyze edges (similarities)
print("\nEdges DataFrame:")
print(edges_df.head())
if len(edges_df) > 0:
    print(f"Average similarity: {edges_df['similarity'].mean():.3f}")

# Use pandas for further analysis
tech_docs = nodes_df[nodes_df['category'] == 'tech']
print(f"\nTech documents: {len(tech_docs)}")
```

## Configuration Options

### SemanticNetwork Parameters

- **metric**: Distance metric for Annoy index ('angular', 'euclidean', etc.) (default: 'angular')
- **n_trees**: Number of trees for Annoy index (more = better accuracy, slower) (default: 10)
- **thresh**: Similarity threshold (0.0 to 1.0) (default: 0.7)
- **top_k**: Maximum neighbors to check per document (default: 100)
- **verbose**: Show progress bars and logging (default: False)

### Method Parameters

- **fit(embeddings, labels=None, ids=None, node_data=None)**: 
  - embeddings are required pre-computed embeddings array with shape (n_docs, embedding_dim)
  - labels are optional text labels/documents for the embeddings
  - ids are optional custom IDs for the embeddings  
  - node_data is optional dictionary containing additional data to attach to nodes
- **transform(thresh=None, top_k=None)**: Optional threshold and top_k overrides
- **fit_transform(embeddings, labels=None, ids=None, node_data=None, thresh=None, top_k=None)**: Combined fit and transform
- **to_pandas(graph)**: Export NetworkX graph to pandas DataFrames

## Performance Tips

- Use `"angular"` metric for cosine similarity (default and recommended)
- Increase `n_trees` for better accuracy (try 50-100 for large datasets)
- Decrease `top_k` if you have memory constraints
- Use smaller embedding models for speed: `"all-MiniLM-L6-v2"`
- Use larger models for accuracy: `"BAAI/bge-large-en-v1.5"`

## Return Values

### transform() and fit_transform() methods

Returns `nx.Graph` - NetworkX graph where:
- Nodes represent documents with attributes including 'name', 'id', and any custom node_data
- Edges represent similarities above the threshold with 'similarity' attribute

### to_pandas() method

Returns `Tuple[pd.DataFrame, pd.DataFrame]` - A tuple containing:
- **nodes**: DataFrame with node attributes (index=node_id, columns include all node attributes)
- **edges**: DataFrame with similarity edges (columns include 'source', 'target', 'similarity', etc.)

## Common Use Cases

- **Document clustering**: Find groups of similar documents
- **Content recommendation**: Identify related content based on semantic similarity
- **Knowledge graph construction**: Build networks of related concepts or entities
- **Duplicate detection**: Find near-duplicate content (though not the primary focus)
- **Exploratory data analysis**: Understand relationships in text collections
- **Search and retrieval**: Build similarity-based search systems

## Requirements

- Python 3.8+
- networkx
- annoy
- numpy
- pandas
- tqdm

Optional for examples:
- sentence-transformers

## Project origin and statement on the use of AI

I love network analysis, and have explored embedding-derived [semantic networks](https://en.wikipedia.org/wiki/Semantic_network) in the past as an alternative approach to representing, clustering and querying news data. 

Whilst using semantic networks for graph analysis on some forthcoming research, I decided to package some of my code for others to use.

I kicked off the project by hand-refactoring my initial code into the class-based structure that forms the core functionality of the current module.

I then used Github Copilot in VSCode to:
- Bootstrap scaffolding, tests, documentation, examples and typing
- Refactor the core methods in the style of the scikit-learn API
- Add additional functionality for convenient analysis of graph structures and to allow the use of custom embeddings.

## Roadmap

Semnet is a relatively simple project focused on core graph construction functionality. Potential future additions:
- Better examples showcasing network analysis on large corpora
- Integration with graph visualization tools
- Performance optimizations for very large datasets

## License

MIT License

## Citation

If you use Semnet in academic work, please cite:

```bibtex
@software{semnet,
  title={Semnet: Semantic Networks from Embeddings},
  author={Ian Goodrich},
  year={2025},
  url={https://github.com/specialprocedures/semnet}
}
```
````

## Quick Start

```python
from semnet import SemanticNetwork
from sentence_transformers import SentenceTransformer  # or any embedding provider

# Your documents
docs = [
    "The cat sat on the mat",
    "A cat was sitting on a mat",  # Similar to first
    "The dog ran in the park", 
    "Python is great for ML",
    "Machine learning with Python"  # Similar to fourth
]

# Generate embeddings (use any embedding method you prefer)
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = embedding_model.encode(docs)

# Optional: provide weights for document importance
weights = [1.0, 0.5, 2.0, 3.0, 2.5]

# Create and configure semantic network (no embedding model needed)
network = SemanticNetwork(
    thresh=0.8,
    verbose=True
)

# Fit and transform documents with your embeddings
representatives = network.fit_transform(embeddings, labels=docs, weights=weights)

# Get statistics
stats = network.get_deduplication_stats()
print(f"Original: {stats['original_count']} documents")
print(f"After deduplication: {stats['deduplicated_count']} documents")
print(f"Reduction: {stats['reduction_ratio']:.1%}")

# Get representative documents
for doc in representatives:
    print(f"- {doc}")
```

## Features

- **Scikit-learn style API** - Familiar fit/transform interface for ML practitioners
- **Bring your own embeddings** - Use any embedding source (OpenAI, Cohere, sentence-transformers, etc.)
- **Blocking for performance** - Only compare documents within specified blocks for massive performance gains
- **Configurable similarity thresholds** - Control how strict the deduplication is
- **Weighted documents** - Give more importance to certain documents when choosing representatives
- **Fast similarity search** - Uses Annoy for efficient approximate nearest neighbor search
- **Graph-based clustering** - Handles transitive similarities (A→B→C all become one group)
- **Verbose mode** - Progress bars and detailed logging
- **Comprehensive statistics** - Detailed information about the deduplication process

## Installation

```bash
pip install semnet
```

For development:

```bash
git clone https://github.com/specialprocedures/semnet.git
cd semnet
pip install -e ".[dev]"
```

## Step-by-Step Usage

### 1. Basic Deduplication

```python
from semnet import SemanticNetwork
from sentence_transformers import SentenceTransformer

# Generate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(your_documents)

# Create network and deduplicate
network = SemanticNetwork(thresh=0.8)
representatives = network.fit_transform(embeddings, labels=your_documents)
```

### 2. Advanced Usage with Weights

```python
from sentence_transformers import SentenceTransformer

# Documents with importance weights
docs = ["doc1", "doc2", "doc3"]
weights = [1.0, 2.0, 0.5]  # doc2 is most important

# Generate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs)

network = SemanticNetwork(
    thresh=0.85,
    verbose=True
)

representatives = network.fit_transform(docs, embeddings, weights=weights)
```

### 3. Separate Fit and Transform

```python
from sentence_transformers import SentenceTransformer

# Generate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs)

network = SemanticNetwork()

# Fit the model on training documents
network.fit(embeddings, labels=docs, weights=weights)

# Transform to get representatives
representatives = network.transform()

# Or get the mapping instead
mapping = network.transform(return_representatives=False)

# Get additional information
stats = network.get_deduplication_stats()
groups = network.get_duplicate_groups()
```

### 4. Using Different Embedding Sources

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Option 1: Use sentence-transformers
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs)

# Option 2: Use OpenAI embeddings (requires openai package)
# import openai
# embeddings = openai.Embedding.create(input=docs, model="text-embedding-ada-002")

# Option 3: Use your own pre-computed embeddings
# custom_embeddings = np.random.rand(len(docs), 384)  # Shape: (n_docs, embedding_dim)

network = SemanticNetwork(thresh=0.8)
representatives = network.fit_transform(embeddings, labels=docs)
```

### 5. Using Blocking for Performance

```python
from sentence_transformers import SentenceTransformer

# Blocking dramatically improves performance by only comparing documents within the same block(s)
docs = ["John Smith", "J. Smith", "Jane Doe", "J. Doe", "Mike Johnson"]

# Generate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs)

# Single blocking variable (e.g., company)
companies = ["TechCorp", "TechCorp", "TechCorp", "TechCorp", "SalesCo"]

network = SemanticNetwork(thresh=0.8)
representatives = network.fit_transform(embeddings, labels=docs, blocks=companies)

# Multiple blocking variables (e.g., company + department)
blocks = [
    ["TechCorp", "Engineering"],
    ["TechCorp", "Engineering"], 
    ["TechCorp", "Product"],
    ["TechCorp", "Product"],
    ["SalesCo", "Sales"]
]

representatives = network.fit_transform(embeddings, labels=docs, blocks=blocks)
```

### 6. Advanced Node Data and IDs

```python
from sentence_transformers import SentenceTransformer

docs = ["Document 1", "Document 2", "Document 3"]
custom_ids = ["doc_001", "doc_002", "doc_003"]

# Additional metadata for each document
node_data = {
    0: {"category": "tech", "priority": 1},
    1: {"category": "science", "priority": 2}, 
    2: {"category": "tech", "priority": 1}
}

# Generate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs)

network = SemanticNetwork(thresh=0.8)
representatives = network.fit_transform(
    embeddings, 
    labels=docs, 
    ids=custom_ids, 
    node_data=node_data
)

# Access additional data through the graph
for node_id in network.graph_.nodes():
    node = network.graph_.nodes[node_id]
    print(f"ID: {node['id']}, Category: {node['category']}, Priority: {node['priority']}")
```

### 7. Exporting to Pandas

```python
from sentence_transformers import SentenceTransformer

docs = ["Document 1", "Document 2", "Document 3"]
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs)

# Add some metadata
node_data = {
    0: {"author": "Alice", "category": "tech"},
    1: {"author": "Bob", "category": "science"},
    2: {"author": "Charlie", "category": "tech"}
}

network = SemanticNetwork(thresh=0.8)
network.fit(embeddings, labels=docs, node_data=node_data)

# Export main graph to pandas DataFrames
nodes_df, edges_df = network.to_pandas()

# Export a subgraph (useful for focused analysis)
largest_component = max(nx.connected_components(network.graph_), key=len)
subgraph = network.graph_.subgraph(largest_component)
sub_nodes_df, sub_edges_df = network.to_pandas(subgraph)

# You can also export any custom NetworkX graph
custom_graph = nx.Graph()
custom_graph.add_node(0, name="Custom Node", attr="value")
custom_nodes_df, custom_edges_df = network.to_pandas(custom_graph)

# Analyze nodes
print("Nodes DataFrame:")
print(nodes_df.head())
print(f"Columns: {list(nodes_df.columns)}")

# Analyze edges (similarities)
print("\nEdges DataFrame:")
print(edges_df.head())
if len(edges_df) > 0:
    print(f"Average similarity: {edges_df['similarity'].mean():.3f}")

# Use pandas for further analysis
tech_docs = nodes_df[nodes_df['category'] == 'tech']
print(f"\nTech documents: {len(tech_docs)}")
```

## Configuration Options

### SemanticNetwork Parameters

- **metric**: Distance metric for Annoy index ('angular', 'euclidean', etc.) (default: 'angular')
- **n_trees**: Number of trees for Annoy index (more = better accuracy, slower) (default: 10)
- **thresh**: Similarity threshold (0.0 to 1.0) (default: 0.7)
- **top_k**: Maximum neighbors to check per document (default: 100)
- **verbose**: Show progress bars and logging (default: False)

### Method Parameters

- **fit(embeddings, labels=None, ids=None, node_data=None, weights=None, blocks=None)**: 
  - embeddings are required pre-computed embeddings array with shape (n_docs, embedding_dim)
  - labels are optional text labels/documents for the embeddings
  - ids are optional custom IDs for the embeddings  
  - node_data is optional dictionary containing additional data to attach to nodes
  - weights are optional importance scores
  - blocks are optional blocking variables (1D list/array or 2D array for multiple variables)
- **transform(X=None, return_representatives=True)**: return_representatives controls output format
- **fit_transform(embeddings, labels=None, ids=None, node_data=None, weights=None, blocks=None, return_representatives=True)**: Combined fit and transform
- **to_pandas(graph=None)**: Export NetworkX graph to pandas DataFrames. Uses fitted graph by default, or accepts custom graph parameter for subgraphs/custom analysis

## Performance Tips

- Use `"angular"` metric for cosine similarity (default)
- Increase `n_trees` for better accuracy (try 50-100 for large datasets)
- Decrease `top_k` if you have memory constraints
- Use smaller embedding models for speed: `"all-MiniLM-L6-v2"`
- Use larger models for accuracy: `"BAAI/bge-large-en-v1.5"`

## Return Values

### transform() and fit_transform() methods

- If `return_representatives=True` (default): `List[str]` - List of unique documents after deduplication
- If `return_representatives=False`: `Dict[int, int]` - Maps document indices to their representatives

### get_deduplication_stats() method

Returns a dictionary with:
- **original_count**: Number of input documents
- **deduplicated_count**: Number of unique documents after deduplication
- **duplicates_found**: Number of documents that were deduplicated
- **reduction_ratio**: Fraction of documents removed (0.0 to 1.0)
- **similarity_pairs**: Number of pairwise similarities found above threshold
- **connected_components**: Number of separate groups in the similarity graph

### get_duplicate_groups() method

Returns `List[List[str]]` - List of groups, where each group contains similar documents

### to_pandas() method

Returns `Tuple[pd.DataFrame, pd.DataFrame]` - A tuple containing:
- **nodes**: DataFrame with node attributes (index=node_id, columns include all node attributes from the graph)
- **edges**: DataFrame with similarity edges (columns include all edge attributes from the graph)

Can be called with:
- `to_pandas()` - Export the fitted semantic network graph
- `to_pandas(custom_graph)` - Export any NetworkX graph (useful for subgraphs)

## Requirements

- Python 3.8+
- sentence-transformers
- annoy
- networkx
- numpy
- pandas
- scikit-learn

## Project origin and statement on the use of AI

I love network analysis, and have explored embedding-derived [semantic networks](https://en.wikipedia.org/wiki/Semantic_network) in the past as an alternative approach to representing, clustering and querying news data. 

Whilst using semantic networks for a complex deduplication task on some forthcoming research, I decided to package some of my code for others to use.

I kicked off the project by hand-refactoring my initial code into the class-based structure that forms the core functionality of the current module.

I then used Github Copilot in VSCode to:
- Bootstrap scaffolding, tests, documentation, examples and typing
- Refactor the core methods in the style of the scikit-learn API
- Add additional functionality for convenient analysis of deduplication outcomes, blocking, and to allow the use of custom embeddings.

## Roadmap
Semnet is a relatively simple project and I don't have plans to add further features. 

This noted, as things currently stand all the examples and tests are AI-generated, and don't showcase use-cases particularly well. Forthcoming additions to this repository will include:
- Better examples and tests, including a demonstration of network analysis on a large corpus and blocking for improved deduplication performance
- Better benchmarking

## License

MIT License

## Citation

If you use Semnet in academic work, please cite:

```bibtex
@software{semnet,
  title={Semnet: Networks from embeddings},
  author={Ian Goodrich},
  year={2025},
  url={https://github.com/specialprocedures/semnet}
}
```