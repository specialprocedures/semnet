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

# Create semantic network
network = SemanticNetwork(
    docs=docs,
    weights=weights,
    embedding_model="BAAI/bge-base-en-v1.5",
    verbose=True
)

# Run deduplication
result = network.deduplicate_documents(thresh=0.8)

print(f"Original: {result['stats']['original_count']} documents")
print(f"After deduplication: {result['stats']['deduplicated_count']} documents")
print(f"Reduction: {result['stats']['reduction_ratio']:.1%}")

# Get representative documents
for doc in result['representatives']:
    print(f"- {doc}")
```

## Features

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

network = SemanticNetwork(docs=your_documents)
result = network.deduplicate_documents(thresh=0.8)
```

### 2. Advanced Usage with Weights

```python
# Documents with importance weights
docs = ["doc1", "doc2", "doc3"]
weights = [1.0, 2.0, 0.5]  # doc2 is most important

network = SemanticNetwork(
    docs=docs, 
    weights=weights,
    embedding_model="all-MiniLM-L6-v2",
    verbose=True
)

result = network.deduplicate_documents(thresh=0.85)
```

### 3. Manual Step-by-Step Process

```python
network = SemanticNetwork(docs=docs)

# Generate embeddings
embeddings = network.embed_documents()

# Build similarity index
index = network.build_vector_index()

# Find similar pairs
similarities = network.get_pairwise_similarities(thresh=0.8, inplace=False)

# Build graph
graph = network.build_graph()

# Get deduplication mapping
mapping = network.get_deduplication_mapping()

# Get duplicate groups
groups = network.get_duplicate_groups()
```

## Configuration Options

- **embedding_model**: Any SentenceTransformer model name or path
- **metric**: Distance metric for Annoy index ('angular', 'euclidean', etc.)
- **n_trees**: Number of trees for Annoy index (more = better accuracy, slower)
- **thresh**: Similarity threshold (0.0 to 1.0)
- **top_k**: Maximum neighbors to check per document
- **verbose**: Show progress bars and logging

## Performance Tips

- Use `"angular"` metric for cosine similarity (default)
- Increase `n_trees` for better accuracy (try 50-100 for large datasets)
- Decrease `top_k` if you have memory constraints
- Use smaller embedding models for speed: `"all-MiniLM-L6-v2"`
- Use larger models for accuracy: `"BAAI/bge-large-en-v1.5"`

## Return Values

The `deduplicate_documents()` method returns a dictionary with:

- **mapping**: `Dict[int, int]` - Maps document indices to their representatives
- **representatives**: `List[str]` - List of unique documents after deduplication
- **similarities**: `pd.DataFrame` - All pairwise similarities above threshold
- **graph**: `nx.Graph` - The similarity graph
- **stats**: `Dict` - Statistics about the deduplication process

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