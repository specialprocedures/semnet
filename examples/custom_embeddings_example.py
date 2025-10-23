#!/usr/bin/env python3
"""
Example of using Semnet with custom embeddings.

This example demonstrates how to provide your own pre-computed embeddings
to Semnet for semantic graph construction. You can use any embedding method:
OpenAI, Cohere, HuggingFace, etc.
"""

import numpy as np
import networkx as nx
from semnet import SemanticNetwork


def main():
    """Run custom embeddings example."""

    # Example documents
    documents = [
        "The cat sat on the mat",
        "A cat was sitting on a mat",  # Very similar to first
        "The dog ran in the park",
        "A dog was running in the park",  # Very similar to third
        "Python is a programming language",
        "Machine learning with Python",  # Somewhat related to fifth
        "I love eating pizza",  # Unrelated
    ]

    print("Custom Embeddings Example")
    print("=" * 50)
    print(f"Input: {len(documents)} documents")
    print()

    # Create custom embeddings (in practice, these might come from a different model)
    # For demonstration, we'll create random embeddings but with some structure
    np.random.seed(42)  # For reproducible results
    embedding_dim = 128

    # Create embeddings where similar documents have similar vectors
    base_embeddings = np.random.rand(len(documents), embedding_dim)

    # Make similar documents more similar by adjusting their embeddings
    # Documents 0 and 1 (cat documents)
    base_embeddings[1] = 0.8 * base_embeddings[0] + 0.2 * base_embeddings[1]

    # Documents 2 and 3 (dog documents)
    base_embeddings[3] = 0.8 * base_embeddings[2] + 0.2 * base_embeddings[3]

    # Documents 4 and 5 (Python documents) - less similar
    base_embeddings[5] = 0.6 * base_embeddings[4] + 0.4 * base_embeddings[5]

    # Normalize embeddings to unit length (common practice)
    custom_embeddings = base_embeddings / np.linalg.norm(
        base_embeddings, axis=1, keepdims=True
    )

    print(f"Created custom embeddings with shape: {custom_embeddings.shape}")
    print()

    # Create semantic network with custom embeddings
    network = SemanticNetwork(
        thresh=0.6,  # Lower threshold to catch our artificial similarities
        verbose=True,
        n_trees=10,
    )

    # Build semantic graph using custom embeddings
    print("Building semantic graph from custom embeddings...")
    graph = network.fit_transform(custom_embeddings, labels=documents)

    print()
    print("SEMANTIC GRAPH RESULTS")
    print("=" * 30)

    # Print basic graph statistics
    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(graph)}")
    print(f"Graph density: {nx.density(graph):.3f}")
    print()

    # Show similarity edges
    print("SIMILARITY EDGES:")
    print("-" * 30)
    if graph.number_of_edges() > 0:
        # Sort edges by similarity (highest first)
        edges_with_sim = [
            (u, v, data["similarity"]) for u, v, data in graph.edges(data=True)
        ]
        edges_with_sim.sort(key=lambda x: x[2], reverse=True)

        for u, v, similarity in edges_with_sim:
            doc1 = graph.nodes[u]["name"]
            doc2 = graph.nodes[v]["name"]
            print(f"  {similarity:.3f}: '{doc1}' <-> '{doc2}'")
    else:
        print("No similarity edges found above the threshold.")
    print()

    # Show connected components (groups of similar documents)
    print("CONNECTED COMPONENTS:")
    print("-" * 30)
    components = list(nx.connected_components(graph))
    components.sort(key=len, reverse=True)  # Largest first

    for i, component in enumerate(components, 1):
        if len(component) > 1:
            print(f"Component {i} ({len(component)} documents):")
            for node in sorted(component):
                doc = graph.nodes[node]["name"]
                print(f"  - {doc}")
            print()
        else:
            # Count isolated nodes
            isolated_count = sum(1 for comp in components if len(comp) == 1)
            if i == 1:  # Only print this once
                print(f"Plus {isolated_count} isolated nodes (no similar documents)")
            break

    # Demonstrate different thresholds
    print("THRESHOLD COMPARISON:")
    print("-" * 30)

    # Try with different thresholds
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        test_graph = network.transform(thresh=thresh)
        print(f"Threshold {thresh}: {test_graph.number_of_edges()} edges")

    print()

    # Export to pandas for analysis
    print("PANDAS EXPORT:")
    print("-" * 30)
    nodes_df, edges_df = network.to_pandas(graph)

    print("Nodes DataFrame:")
    print(nodes_df[["name", "id"]].head())
    print()

    if len(edges_df) > 0:
        print("Edges DataFrame:")
        print(edges_df[["source", "target", "similarity"]].head())
        print(f"\nSimilarity statistics:")
        print(f"  Mean: {edges_df['similarity'].mean():.3f}")
        print(f"  Max: {edges_df['similarity'].max():.3f}")
        print(f"  Min: {edges_df['similarity'].min():.3f}")
    else:
        print("No edges to analyze.")

    print()
    print("Custom embeddings example complete!")
    print("\nThis demonstrates how you can:")
    print("- Use any embedding source (OpenAI, Cohere, custom models)")
    print("- Build semantic graphs from pre-computed embeddings")
    print("- Analyze relationships and similarity patterns")
    print("- Export results for further analysis")


if __name__ == "__main__":
    main()
