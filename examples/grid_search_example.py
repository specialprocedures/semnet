#!/usr/bin/env python3
"""
Grid search example for SemanticNetwork parameter optimization.

This example shows how to find optimal parameters for minimizing orphan nodes
and edge count using a manual grid search approach.
"""

import numpy as np
import networkx as nx
from itertools import product

# Import semnet - adjust path as needed
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from semnet import SemanticNetwork


def evaluate_parameters(embeddings, documents, metric, n_trees, thresh, top_k):
    """
    Evaluate a parameter combination and return graph metrics.
    """
    network = SemanticNetwork(
        metric=metric, n_trees=n_trees, thresh=thresh, top_k=top_k, verbose=False
    )

    # Build the graph
    graph = network.fit_transform(embeddings, labels=documents)

    # Calculate metrics
    orphan_count = sum(1 for node in graph.nodes() if len(list(graph.neighbors(node))) == 0)
    edge_count = graph.number_of_edges()
    node_count = graph.number_of_nodes()
    components = nx.number_connected_components(graph)

    return {
        "graph": graph,
        "orphans": orphan_count,
        "edges": edge_count,
        "nodes": node_count,
        "components": components,
        "orphan_ratio": orphan_count / node_count if node_count > 0 else 0,
        "edge_density": edge_count / (node_count * (node_count - 1) / 2) if node_count > 1 else 0,
    }


def main():
    """Run manual grid search example."""

    print("Manual Grid Search for SemanticNetwork Parameter Optimization")
    print("=" * 65)
    print()

    # Generate clustered random vectors to ensure some similarities
    print("Generating clustered random vectors...")
    np.random.seed(42)  # For reproducible results
    n_vectors = 1000
    embedding_dim = 384  # Common embedding dimension
    n_clusters = 20  # Number of clusters to create

    # Create cluster centers
    cluster_centers = np.random.randn(n_clusters, embedding_dim)
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)

    # Assign each vector to a cluster and add noise
    embeddings = []
    for i in range(n_vectors):
        cluster_id = i % n_clusters  # Distribute evenly across clusters
        center = cluster_centers[cluster_id]

        # Add noise to the cluster center (smaller noise = more similar vectors)
        noise = np.random.randn(embedding_dim) * 0.3  # Adjust noise level
        vector = center + noise

        # Normalize to unit length
        vector = vector / np.linalg.norm(vector)
        embeddings.append(vector)

    embeddings = np.array(embeddings)

    # Create dummy document labels
    documents = [f"document_{i:04d}" for i in range(n_vectors)]

    print(f"Generated {n_vectors} random vectors with dimension {embedding_dim}")
    print(f"Embeddings shape: {embeddings.shape}")
    print()

    # Define parameter grid with lower thresholds for random vectors
    param_grid = {
        "thresh": [0.01, 0.05, 0.1, 0.2],  # Lower similarity thresholds for random data
        "top_k": [5, 10, 20],  # Max neighbors to check
        "metric": ["angular", "euclidean"],  # Distance metrics
    }

    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print()

    # Run grid search
    print("Running parameter evaluation...")
    results = []

    # Generate all parameter combinations
    param_combinations = list(
        product(
            param_grid["metric"],
            [10],  # Fixed n_trees for speed
            param_grid["thresh"],
            param_grid["top_k"],
        )
    )

    print(f"Evaluating {len(param_combinations)} parameter combinations...")
    print()

    for i, (metric, n_trees, thresh, top_k) in enumerate(param_combinations, 1):
        print(f"[{i:2d}/{len(param_combinations)}] metric={metric}, thresh={thresh}, top_k={top_k}")

        try:
            metrics = evaluate_parameters(embeddings, documents, metric, n_trees, thresh, top_k)

            result = {
                "metric": metric,
                "n_trees": n_trees,
                "thresh": thresh,
                "top_k": top_k,
                **metrics,
            }
            results.append(result)

            print(
                f"         → orphans: {metrics['orphans']}, edges: {metrics['edges']}, components: {metrics['components']}"
            )

        except Exception as e:
            print(f"         → ERROR: {e}")

    print()
    print("OPTIMIZATION RESULTS")
    print("=" * 50)

    if not results:
        print("No successful parameter combinations found!")
        return

    # Find best parameters for different objectives
    objectives = {
        "minimize_orphans": lambda r: (
            r["orphans"],
            r["edges"],
        ),  # Primary: min orphans, secondary: min edges
        "minimize_edges": lambda r: (
            r["edges"],
            r["orphans"],
        ),  # Primary: min edges, secondary: min orphans
        "balanced": lambda r: (r["orphans"] * 2 + r["edges"]),  # Weighted combination
        "maximize_connectivity": lambda r: (
            -r["components"],
            r["orphans"],
        ),  # Fewer components, fewer orphans
    }

    for obj_name, obj_func in objectives.items():
        print(f"\n{obj_name.upper()}:")
        print("-" * (len(obj_name) + 1))

        best_result = min(results, key=obj_func)

        print(f"Best parameters:")
        print(f"  - metric: {best_result['metric']}")
        print(f"  - thresh: {best_result['thresh']}")
        print(f"  - top_k: {best_result['top_k']}")
        print(f"Resulting graph:")
        print(f"  - Nodes: {best_result['nodes']}")
        print(f"  - Edges: {best_result['edges']}")
        print(f"  - Orphan nodes: {best_result['orphans']} ({best_result['orphan_ratio']:.1%})")
        print(f"  - Connected components: {best_result['components']}")
        print(f"  - Edge density: {best_result['edge_density']:.3f}")

    print()
    print("PARAMETER ANALYSIS")
    print("=" * 40)

    # Analyze parameter effects
    print("\nThreshold effects:")
    thresh_effects = {}
    for result in results:
        thresh = result["thresh"]
        if thresh not in thresh_effects:
            thresh_effects[thresh] = {"orphans": [], "edges": []}
        thresh_effects[thresh]["orphans"].append(result["orphans"])
        thresh_effects[thresh]["edges"].append(result["edges"])

    for thresh in sorted(thresh_effects.keys()):
        avg_orphans = np.mean(thresh_effects[thresh]["orphans"])
        avg_edges = np.mean(thresh_effects[thresh]["edges"])
        print(f"  thresh={thresh}: avg_orphans={avg_orphans:.1f}, avg_edges={avg_edges:.1f}")

    print("\nTop-k effects:")
    topk_effects = {}
    for result in results:
        top_k = result["top_k"]
        if top_k not in topk_effects:
            topk_effects[top_k] = {"orphans": [], "edges": []}
        topk_effects[top_k]["orphans"].append(result["orphans"])
        topk_effects[top_k]["edges"].append(result["edges"])

    for top_k in sorted(topk_effects.keys()):
        avg_orphans = np.mean(topk_effects[top_k]["orphans"])
        avg_edges = np.mean(topk_effects[top_k]["edges"])
        print(f"  top_k={top_k}: avg_orphans={avg_orphans:.1f}, avg_edges={avg_edges:.1f}")

    print()
    print("KEY INSIGHTS:")
    print("- Lower threshold → fewer orphans but more edges")
    print("- Higher top_k → more connection opportunities")
    print("- Angular metric typically works well for text embeddings")
    print("- Balance orphans vs edges based on your use case")


if __name__ == "__main__":
    main()
