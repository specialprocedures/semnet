"""
Example: Exporting Semantic Networks to Pandas

This example demonstrates how to use the to_pandas() method to export
the semantic network graph structure to pandas DataFrames for further
analysis and visualization.
"""

import numpy as np
import pandas as pd
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

# Document importance weights
weights = [1.0, 0.8, 1.5, 1.2, 2.0, 1.8, 1.0]

print("=== Semantic Network Pandas Export Example ===\n")

# Create and fit the semantic network
network = SemanticNetwork(
    thresh=0.85, verbose=True  # High threshold to ensure quality matches
)

print("Fitting semantic network...")
network.fit(
    embeddings=embeddings,
    labels=docs,
    ids=custom_ids,
    node_data=node_data,
    weights=weights,
)

# Get basic statistics
stats = network.get_deduplication_stats()
print(f"\nDeduplication Statistics:")
print(f"- Original documents: {stats['original_count']}")
print(f"- After deduplication: {stats['deduplicated_count']}")
print(f"- Duplicates found: {stats['duplicates_found']}")
print(f"- Reduction ratio: {stats['reduction_ratio']:.1%}")
print(f"- Similarity pairs: {stats['similarity_pairs']}")

# Export to pandas DataFrames
print("\n=== Exporting to Pandas DataFrames ===")
nodes_df, edges_df = network.to_pandas()

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

# Find highest weighted documents
print("\nTop 3 documents by weight:")
top_weighted = nodes_df.nlargest(3, "weight")[["name", "weight", "topic"]]
print(top_weighted)

# Analyze sentiment distribution
print("\nSentiment distribution:")
sentiment_counts = nodes_df["sentiment"].value_counts()
print(sentiment_counts)

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

# Get duplicate groups
groups = network.get_duplicate_groups()
if groups:
    print(f"\n=== Duplicate Groups Found ===")
    for i, group in enumerate(groups, 1):
        print(f"\nGroup {i} ({len(group)} documents):")
        for doc in group:
            print(f"  - {doc}")
else:
    print("\nNo duplicate groups found (all documents are unique)")

print("\n=== Example Complete ===")
print("\nYou can now use the nodes_df and edges_df DataFrames for:")
print("- Further analysis with pandas")
print("- Visualization with matplotlib/seaborn")
print("- Export to CSV, Excel, or databases")
print("- Integration with other data science workflows")
