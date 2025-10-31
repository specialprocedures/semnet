# Semnet: efficient graph structures from embeddings

![Embeddings of Guardian headlines represented as a network structure by Semnet and visualised by Cosmograph](img/cosmo_semnet.png)
_Embeddings of Guardian headlines represented as a network by Semnet and visualised in [Cosmograph](cosmograph.app)_

## Introduction
Semnet constructs graph structures from embeddings, enabling graph-based analysis and operations over collections of embedded documents.

Semnet uses [Annoy](https://github.com/spotify/annoy) to perform efficient pair-wise distance calculations, allowing for million-embedding network construction in under ten minutes on consumer hardware.

Graphs are returned as [NetworkX](https://networkx.org) objects, opening up a wide range of algorithms for downstream use.

The name "Semnet" derives from _[semantic network](https://en.wikipedia.org/wiki/Semantic_network)_[^1], as it was initially designed for an NLP use-case, but the tool will work well with any form of embedded document (e.g., images, audio, even or [graphs](https://arxiv.org/abs/1707.05005)).

[^1]: Technically-speaking a [Semantic Similarity Network (SSN)](https://en.wikipedia.org/wiki/Semantic_similarity_network)

Semnet may be used for:
- **Graph algorithms**: enrich your data with [communities](https://networkx.org/documentation/stable/reference/algorithms/community.html), [centrality](https://networkx.org/documentation/stable/reference/algorithms/centrality.html) and [much more](https://networkx.org/documentation/stable/reference/algorithms/) for down-stream use in search, RAG and context engineering 
- **Deduplication**: remove duplicate records (e.g., "Donald Trump", "Donald J. Trump) from datasets
- **Exploratory data analysis and visualisation**, [Cosmograph](https://cosmograph.app/) works brilliantly for large corpora

Exposing the full NetworkX and Annoy APIs, Semnet offers plenty of opportunity for experimentation depending on your use-case. 

Check out the [launch blog](https://igdr.ch/posts/semnet-intro/) for more about Semnet and the [examples](https://igdr.ch/posts/semnet-examples/) for inspiration.

## Installation

```bash
pip install semnet
```
## Quick Start
```python
from semnet import SemanticNetwork
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
sem = SemanticNetwork(thresh=0.3, verbose=True)  # Larger values give sparser networks

# Build a NetworkX graph object from your embeddings
G = sem.fit_transform(embeddings, labels=docs)

# Export to pandas using the standalone function
from semnet import to_pandas
nodes, edges = to_pandas(G)
```

## Requirements

- Python 3.8+
- networkx
- annoy
- numpy
- pandas
- tqdm

Recommended for examples:
- sentence-transformers
- cosmograph

## Project origin

I love network analysis, and have explored embedding-derived [semantic networks](https://en.wikipedia.org/wiki/Semantic_network) in the past as an alternative approach to representing, clustering and querying news data. 

Semnet started life as a few functions I'd been using for deduplication for a forthcoming piece of research. I could see a number of potential uses for my code, so I decided to package it up for others to use.

## Statement on the use of AI

I kicked off the project by hand-refactoring my initial code into the class-based structure that forms the core functionality of the current module.

I then used Github Copilot in VSCode to:
- Bootstrap scaffolding, tests, documentation, examples and typing
- Refactor the core methods in the style of the scikit-learn API
- Add additional functionality, e.g., the ability to pass custom data to nodes
- Walk me through deployment to [readthedocs](https://semnetdocs.readthedocs.io/) and [pypi](https://pypi.org/project/semnet/)

## Roadmap

Semnet is a relatively simple project focused on core graph construction functionality. I don't have much in the way of immediate plans to expand it, however can see the potential for a few future additions: 

- Performance optimizations for very large datasets
- Utilities for deduplication, as that's my main use case 
- Integration with graph visualization tools

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
