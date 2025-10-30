# Semnet: Semantic networks from embeddings

**Semnet efficiently constructs network structures from embeddings, enabling graph-based analysis and operations over embedded documents, images, and more.**

The name "Semnet" derives from _[semantic network](https://en.wikipedia.org/wiki/Semantic_network)_, as it was initially designed for an NLP use-case, but it will work well with any form of embedded document (e.g., images, audio, even or [graphs](https://arxiv.org/abs/1707.05005)).

## Features

Semnet is a relatively small library, which does just one thing: efficiently construct network structures from embeddings.

Its key features are:

- Rapid conversion of large embedding collections to graph format, with nodes as documents and edges as document similarity.
- It's fast and memory efficient: Semnet uses [Annoy](https://github.com/spotify/annoy) under the hood to perform efficient pair-wise distance calculations, allowing for million-node networks to be constructed in minutes on consumer hardware.
- Graphs are returned as [NetworkX](https://networkx.org) objects, opening up a wide range of algorithms for your project.
- Pass arbitrary metadata during graph construction or update it later in NetworkX.
- Control graph construction by setting distance measures, similarity cut-offs, and limits on outbound edges per node.
- Easily convert to [pandas](https://pandas.pydata.org/) for downstream use.

## Use cases

Semnet may be used for:

- **Graph algorithms**: enrich your data with [communities](https://networkx.org/documentation/stable/reference/algorithms/community.html), [centrality](https://networkx.org/documentation/stable/reference/algorithms/centrality.html) and [much more](https://networkx.org/documentation/stable/reference/algorithms/) for use in NLP, search, RAG and context engineering.
- **Deduplication**: remove duplicate records (e.g., "Donald Trump", "Donald J. Trump") from datasets.
- **Exploratory data analysis and visualisation**, [Cosmograph](https://cosmograph.app/) works brilliantly for large corpora.

## Quick start

You can easily install Semnet with `pip`.

```bash
pip install semnet
```

All you need to start building your network is a set of embeddings and (optionally) some labels.

```python
from semnet import SemanticNetwork, to_pandas
from sentence_transformers import SentenceTransformer

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

# Export to pandas
nodes_df, edges_df = to_pandas(G)
```

## What problem does Semnet solve?

Graph construction entails finding pairwise relationships (edges) between entities (nodes) in a dataset.

For large corpora, scaling problems rapidly become apparent as the number of possible pairs in a set scales quadratically.

### Naive approach

If we were to naively attempt to construct a graph from a modestly-sized set of documents we encounter problems early on with modestly-sized corpora. Building a graph from 10,000 documents would entail operations across 50 million pairs, for 100,000 it's around 5 billion!

Iterating over each pair is very slow, and faster approaches via [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html) run into memory problems:

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

### With Semnet

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

## Why use graph structures?

By opening up the NetworkX API to embedded documents, Semnet provides a new suite of tools and metrics for workflows in domains such as NLP, RAG, search and context engineering.

For most use cases, Semnet will work best as a complement to traditional spatial workflows, rather than as a replacement. Its power lies in encoding information about relationships between data points, which can be used as features in downstream tasks.

Approaches will vary depending on your use case, but benefits include:

- Accessing network structures like paths, local neighbourhoods and subgraphs
- Using centrality measures and communities that capture relationships and structure
- Making some very pretty visualisations

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
quickstart
api
examples
parameters
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`