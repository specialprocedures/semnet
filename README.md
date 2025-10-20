# Semnet: Semantic Network Deduplication

A Python package for building semantic networks using embeddings and graph clustering to perform intelligent deduplication of text data.

## Overview

Semnet combines three powerful techniques for semantic deduplication:

1. **Sentence Embeddings** - Uses pre-trained transformer models to create dense vector representations of text
2. **Approximate Nearest Neighbor Search** - Efficiently finds similar documents using Annoy indexing
3. **Graph Clustering** - Builds similarity graphs and finds connected components to group duplicates

## Quick Start

```python
from semnet import SemanticNetwork

# Your documents
docs = [
    "The cat sat on the mat",
    "A cat was sitting on a mat",  # Similar to first
    "The dog ran in the park", 
    "Python is great for ML",
    "Machine learning with Python"  # Similar to fourth
]

# Optional: provide weights for document importance
weights = [1.0, 0.5, 2.0, 3.0, 2.5]

# Create and configure semantic network
network = SemanticNetwork(
    embedding_model="BAAI/bge-base-en-v1.5",
    thresh=0.8,
    verbose=True
)

# Fit and transform documents
representatives = network.fit_transform(docs, weights=weights)

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
- **Multiple embedding models** - Support for any SentenceTransformer model
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

network = SemanticNetwork(thresh=0.8)
representatives = network.fit_transform(your_documents)
```

### 2. Advanced Usage with Weights

```python
# Documents with importance weights
docs = ["doc1", "doc2", "doc3"]
weights = [1.0, 2.0, 0.5]  # doc2 is most important

network = SemanticNetwork(
    embedding_model="all-MiniLM-L6-v2",
    thresh=0.85,
    verbose=True
)

representatives = network.fit_transform(docs, weights=weights)
```

### 3. Separate Fit and Transform

```python
network = SemanticNetwork()

# Fit the model on training documents
network.fit(docs, weights=weights)

# Transform to get representatives
representatives = network.transform()

# Or get the mapping instead
mapping = network.transform(return_representatives=False)

# Get additional information
stats = network.get_deduplication_stats()
groups = network.get_duplicate_groups()
```

## Configuration Options

### SemanticNetwork Parameters

- **embedding_model**: Any SentenceTransformer model name or path (default: "BAAI/bge-base-en-v1.5")
- **metric**: Distance metric for Annoy index ('angular', 'euclidean', etc.) (default: 'angular')
- **n_trees**: Number of trees for Annoy index (more = better accuracy, slower) (default: 10)
- **thresh**: Similarity threshold (0.0 to 1.0) (default: 0.7)
- **top_k**: Maximum neighbors to check per document (default: 100)
- **verbose**: Show progress bars and logging (default: False)

### Method Parameters

- **fit(X, y=None, weights=None)**: X is list of documents, weights are optional importance scores
- **transform(X=None, return_representatives=True)**: return_representatives controls output format
- **fit_transform(X, y=None, weights=None, return_representatives=True)**: Combined fit and transform

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

## Requirements

- Python 3.8+
- sentence-transformers
- annoy
- networkx
- numpy
- pandas
- scikit-learn

## License

MIT License

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Citation

If you use Semnet in academic work, please cite:

```bibtex
@software{semnet,
  title={Semnet: Semantic Network Deduplication},
  author={Ian Goodrich},
  year={2025},
  url={https://github.com/specialprocedures/semnet}
}
```