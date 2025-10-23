"""
Example: Exporting Semantic Networks to Pandas

This example demonstrates how to use the to_pandas() method to export
semantic network graph structures to pandas DataFrames for further
analysis and visualization.
"""

import numpy as np
import pandas as pd
import networkx as nx
from semnet import SemanticNetwork

# Create sample documents
docs = [
    "The cat sat on the mat",
    "A cat was sitting on a mat",  # Similar to first
    "The dog ran in the park",
    "A dog was running in the park",  # Similar to third
    "Python is great for machine learning",
    "Machine learning with Python is powerful",  # Similar to fifth
    "The weather is nice today",  # Standalone document
]

# Generate some embeddings (in practice, use a real embedding model)
np.random.seed(42)  # For reproducibility
embeddings = []

# Create similar embeddings for similar documents
base1 = np.random.rand(128)  # Base for cat documents
base2 = np.random.rand(128)  # Base for dog documents
base3 = np.random.rand(128)  # Base for Python/ML documents
base4 = np.random.rand(128)  # Base for weather document

embeddings.extend(
    [
        base1 + 0.01 * np.random.rand(128),  # Cat 1
        base1 + 0.01 * np.random.rand(128),  # Cat 2 (similar)
        base2 + 0.01 * np.random.rand(128),  # Dog 1
        base2 + 0.01 * np.random.rand(128),  # Dog 2 (similar)
        base3 + 0.01 * np.random.rand(128),  # Python 1
        base3 + 0.01 * np.random.rand(128),  # Python 2 (similar)
        base4 + 0.01 * np.random.rand(128),  # Weather (standalone)
    ]
)

embeddings = np.array(embeddings)
# Normalize for cosine similarity
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Add some metadata for each document
node_data = {
    0: {"topic": "animals", "sentiment": "neutral", "length": "short"},
    1: {"topic": "animals", "sentiment": "neutral", "length": "short"},
    2: {"topic": "animals", "sentiment": "positive", "length": "short"},
    3: {"topic": "animals", "sentiment": "positive", "length": "short"},
    4: {"topic": "technology", "sentiment": "positive", "length": "medium"},
    5: {"topic": "technology", "sentiment": "positive", "length": "medium"},
    6: {"topic": "weather", "sentiment": "positive", "length": "short"},
}

# Custom document IDs
custom_ids = [f"doc_{i:03d}" for i in range(len(docs))]

print("=== Semantic Network Pandas Export Example ===\n")

# Create and build the semantic network
network = SemanticNetwork(
    thresh=0.85, verbose=True  # High threshold to ensure quality matches
)

print("Building semantic network...")
graph = network.fit_transform(
    embeddings=embeddings,
    labels=docs,
    ids=custom_ids,
    node_data=node_data,
)

# Get basic graph statistics
print(f"\nGraph Statistics:")
print(f"- Nodes: {graph.number_of_nodes()}")
print(f"- Edges: {graph.number_of_edges()}")
print(f"- Connected components: {nx.number_connected_components(graph)}")
print(f"- Density: {nx.density(graph):.3f}")

# Export to pandas DataFrames
print("\n=== Exporting to Pandas DataFrames ===")
nodes_df, edges_df = network.to_pandas(graph)

print(f"\nNodes DataFrame shape: {nodes_df.shape}")
print(f"Columns: {list(nodes_df.columns)}")
print("\nFirst few nodes:")
print(nodes_df.head())

print(f"\nEdges DataFrame shape: {edges_df.shape}")
if len(edges_df) > 0:
    print(f"Columns: {list(edges_df.columns)}")
    print("\nFirst few edges:")
    print(edges_df.head())
    print(f"\nSimilarity statistics:")
    print(f"- Min similarity: {edges_df['similarity'].min():.3f}")
    print(f"- Max similarity: {edges_df['similarity'].max():.3f}")
    print(f"- Mean similarity: {edges_df['similarity'].mean():.3f}")
else:
    print("No edges found (no similarities above threshold)")

# Analyze the data using pandas
print("\n=== Pandas Analysis Examples ===")

# Group by topic
print("\nDocuments by topic:")
topic_counts = nodes_df["topic"].value_counts()
print(topic_counts)

# Analyze sentiment distribution
print("\nSentiment distribution:")
sentiment_counts = nodes_df["sentiment"].value_counts()
print(sentiment_counts)

# Find documents by length
print("\nDocuments by length:")
length_counts = nodes_df["length"].value_counts()
print(length_counts)

# Find connected documents (if any edges exist)
if len(edges_df) > 0:
    print("\nSimilarity connections found:")
    for _, edge in edges_df.iterrows():
        source_name = nodes_df.loc[edge["source"], "name"]
        target_name = nodes_df.loc[edge["target"], "name"]
        similarity = edge["similarity"]
        print(
            f"- {source_name[:30]}... ↔ {target_name[:30]}... (sim: {similarity:.3f})"
        )

# Analyze connected components
components = list(nx.connected_components(graph))
components.sort(key=len, reverse=True)

if any(len(comp) > 1 for comp in components):
    print(f"\n=== Connected Components (Similar Document Groups) ===")
    for i, component in enumerate(components, 1):
        if len(component) > 1:
            print(f"\nComponent {i} ({len(component)} documents):")
            for node in sorted(component):
                doc_name = graph.nodes[node]["name"]
                doc_topic = graph.nodes[node]["topic"]
                print(f"  - {doc_name} (topic: {doc_topic})")
else:
    print("\nNo connected components found (all documents are isolated)")

# Demonstrate filtering and analysis
print("\n=== Advanced Pandas Analysis ===")

# Filter nodes by topic
tech_docs = nodes_df[nodes_df["topic"] == "technology"]
print(f"\nTechnology documents ({len(tech_docs)}):")
for _, row in tech_docs.iterrows():
    print(f"  - {row['name']}")

# Create a summary table
print("\nSummary by topic:")
summary = (
    nodes_df.groupby("topic")
    .agg(
        {
            "name": "count",
            "sentiment": lambda x: x.mode().iloc[0] if not x.empty else "unknown",
        }
    )
    .rename(columns={"name": "doc_count", "sentiment": "common_sentiment"})
)
print(summary)

# If edges exist, analyze similarity patterns
if len(edges_df) > 0:
    print("\nSimilarity network analysis:")

    # Join edges with node data
    edges_enriched = edges_df.merge(
        nodes_df[["topic"]],
        left_on="source",
        right_index=True,
        suffixes=("", "_source"),
    ).merge(
        nodes_df[["topic"]],
        left_on="target",
        right_index=True,
        suffixes=("", "_target"),
    )

    # Check if similarities are within topics or across topics
    edges_enriched["same_topic"] = (
        edges_enriched["topic"] == edges_enriched["topic_target"]
    )

    print(f"  - Within-topic connections: {edges_enriched['same_topic'].sum()}")
    print(f"  - Cross-topic connections: {(~edges_enriched['same_topic']).sum()}")

# Demonstrate subgraph analysis
if graph.number_of_edges() > 0:
    print("\n=== Subgraph Analysis ===")

    # Get largest connected component
    largest_component = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_component)

    print(f"Largest connected component:")
    print(f"  - Nodes: {subgraph.number_of_nodes()}")
    print(f"  - Edges: {subgraph.number_of_edges()}")

    # Export subgraph
    sub_nodes_df, sub_edges_df = network.to_pandas(subgraph)
    print(f"  - Subgraph nodes DataFrame shape: {sub_nodes_df.shape}")
    print(f"  - Subgraph edges DataFrame shape: {sub_edges_df.shape}")

print("\n=== Example Complete ===")
print("\nYou can now use the nodes_df and edges_df DataFrames for:")
print("- Further analysis with pandas")
print("- Visualization with matplotlib/seaborn/plotly")
print("- Network analysis with NetworkX")
print("- Export to CSV, Excel, or databases")
print("- Integration with other data science workflows")
print("- Machine learning on graph features")
