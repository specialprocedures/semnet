#!/usr/bin/env python3
"""
Basic example of using Semnet for semantic graph construction.

This example shows how to build semantic networks from document embeddings
using approximate nearest neighbor search and graph construction. Users must provide
their own embeddings (e.g., from sentence-transformers, OpenAI, etc.)
"""

from semnet import SemanticNetwork, to_pandas
from sentence_transformers import SentenceTransformer
import numpy as np
import networkx as nx


def main():
    """Run basic semantic graph construction example."""

    # Example documents with some semantic relationships
    documents = [
        "The cat sat on the mat",
        "A cat was sitting on a mat",  # Very similar to first
        "The feline was on the rug",  # Somewhat similar to first
        "The dog ran in the park",
        "A dog was running in the park",  # Very similar to fourth
        "Python is a programming language",
        "Python programming language",  # Similar to sixth
        "Machine learning with Python",  # Somewhat related to sixth
        "The weather is nice today",
        "Today the weather is beautiful",  # Similar to ninth
        "I love eating pizza",  # Unrelated
    ]

    print("Starting semantic graph construction...")
    print(f"Input: {len(documents)} documents")
    print()

    # Generate embeddings using sentence-transformers (users can use any embedding method)
    print("Generating embeddings...")
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )  # Fast, good quality model
    embeddings = embedding_model.encode(documents)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    print()

    # Create semantic network - no embedding model needed since we provide embeddings
    network = SemanticNetwork(
        verbose=True,  # Show progress bars and detailed info
        n_trees=10,  # Good balance of speed/accuracy
        thresh=0.25,  # Similarity threshold (0.0-1.0)
        top_k=5,  # Max neighbors to check per document
    )

    # Fit the model with provided embeddings and get the semantic graph
    print("Building semantic graph...")
    graph = network.fit_transform(embeddings, labels=documents)

    print()
    print("=" * 60)
    print("SEMANTIC GRAPH RESULTS")
    print("=" * 60)

    # Print basic graph statistics
    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(graph)}")
    print(f"Average clustering coefficient: {nx.average_clustering(graph):.3f}")
    print()

    # Show sample of nodes with their attributes
    print("SAMPLE NODES (with attributes):")
    print("-" * 40)
    for i, (node, data) in enumerate(list(graph.nodes(data=True))[:5]):
        print(f"Node {node}: {data['label'][:50]}...")
        print(f"  Attributes: {dict(data)}")
        print()

    # Show edges (similarities) above threshold
    print("SIMILARITY EDGES:")
    print("-" * 40)
    if graph.number_of_edges() > 0:
        # Sort edges by similarity (highest first)
        edges_with_sim = [
            (u, v, data[\"weight\"]) for u, v, data in graph.edges(data=True)
        ]
        edges_with_sim.sort(key=lambda x: x[2], reverse=True)

        print(f"Found {len(edges_with_sim)} similarity connections:")
        for u, v, similarity in edges_with_sim[:10]:  # Show top 10
            doc1 = graph.nodes[u]["label"]
            doc2 = graph.nodes[v]["label"]
            print(f"  {similarity:.3f}: '{doc1[:40]}...' <-> '{doc2[:40]}...'")

        if len(edges_with_sim) > 10:
            print(f"  ... and {len(edges_with_sim) - 10} more edges")
    else:
        print("No similarity edges found above the threshold.")
    print()

    # Show connected components (groups of similar documents)
    print("CONNECTED COMPONENTS:")
    print("-" * 40)
    components = list(nx.connected_components(graph))
    components.sort(key=len, reverse=True)  # Largest first

    for i, component in enumerate(components, 1):
        if len(component) > 1:
            print(f"Component {i} ({len(component)} documents):")
            for node in sorted(component):
                doc = graph.nodes[node]["label"]
                print(f"  - {doc}")
            print()
        else:
            # Count isolated nodes
            isolated_count = sum(1 for comp in components if len(comp) == 1)
            if i == 1:  # Only print this once
                print(f"Plus {isolated_count} isolated nodes (no similar documents)")
            break

    # Export to pandas for further analysis
    print("PANDAS EXPORT EXAMPLE:")
    print("-" * 40)
    nodes_df, edges_df = to_pandas(graph)

    print("Nodes DataFrame:")
    print(nodes_df[["label", "node_id"]].head())
    print()

    if len(edges_df) > 0:
        print("Edges DataFrame:")
        print(edges_df[["source", "target", "similarity"]].head())
        print(f"\nEdge statistics:")
        print(f"  Mean similarity: {edges_df['similarity'].mean():.3f}")
        print(f"  Max similarity: {edges_df['similarity'].max():.3f}")
        print(f"  Min similarity: {edges_df['similarity'].min():.3f}")
    else:
        print("No edges to export.")
    print()

    # Demonstrate custom thresholds
    print("CUSTOM THRESHOLD EXAMPLE:")
    print("-" * 40)

    # Try with a higher threshold
    high_thresh_graph = network.transform(thresh=0.5)
    print(f"With threshold 0.5: {high_thresh_graph.number_of_edges()} edges")

    # Try with a lower threshold
    low_thresh_graph = network.transform(thresh=0.1)
    print(f"With threshold 0.1: {low_thresh_graph.number_of_edges()} edges")
    print()

    print("Graph construction complete!")
    print("\nNext steps:")
    print("- Analyze the graph with NetworkX algorithms")
    print("- Export to formats like GraphML, GML, or JSON")
    print("- Visualize with tools like matplotlib, plotly, or networkx")
    print("- Use for downstream tasks like clustering, search, or recommendation")


if __name__ == "__main__":
    main()
