#!/usr/bin/env python3
"""
Basic example of using Semnet for document deduplication.

This example shows how to deduplicate a list of similar documents
using semantic embeddings and graph clustering.
"""

from semnet import SemanticNetwork


def main():
    """Run basic deduplication example."""

    # Example documents with some duplicates
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

    # Optional: provide weights based on document importance/frequency
    # Higher weights = more likely to be chosen as representative
    weights = [
        2.0,  # "The cat sat on the mat" - prefer this version
        1.0,  # "A cat was sitting on a mat"
        0.5,  # "The feline was on the rug" - less preferred
        1.5,  # "The dog ran in the park"
        1.0,  # "A dog was running in the park"
        3.0,  # "Python is a programming language" - highly preferred
        2.0,  # "Python programming language"
        1.0,  # "Machine learning with Python"
        1.0,  # "The weather is nice today"
        2.0,  # "Today the weather is beautiful" - prefer this version
        1.0,  # "I love eating pizza"
    ]

    print("🔍 Starting semantic deduplication...")
    print(f"📄 Input: {len(documents)} documents")
    print()

    # Create semantic network with verbose output
    network = SemanticNetwork(
        docs=documents,
        weights=weights,
        embedding_model="all-MiniLM-L6-v2",  # Fast, good quality model
        verbose=True,  # Show progress bars and detailed info
        n_trees=10,  # Good balance of speed/accuracy
    )

    # Run the full deduplication pipeline
    result = network.deduplicate_documents(
        thresh=0.75,  # Similarity threshold (0.0-1.0)
        top_k=50,  # Max neighbors to check per document
    )

    print()
    print("=" * 60)
    print("📊 DEDUPLICATION RESULTS")
    print("=" * 60)

    # Print statistics
    stats = result["stats"]
    print(f"📄 Original documents: {stats['original_count']}")
    print(f"✨ After deduplication: {stats['deduplicated_count']}")
    print(f"🗑️  Duplicates removed: {stats['duplicates_found']}")
    print(f"📉 Reduction ratio: {stats['reduction_ratio']:.1%}")
    print(f"🔗 Similarity pairs found: {stats['similarity_pairs']}")
    print(f"🌐 Connected components: {stats['connected_components']}")
    print()

    # Show representative documents
    print("📋 REPRESENTATIVE DOCUMENTS:")
    print("-" * 40)
    for i, doc in enumerate(result["representatives"], 1):
        print(f"{i:2d}. {doc}")
    print()

    # Show duplicate groups
    print("👥 DUPLICATE GROUPS:")
    print("-" * 40)

    groups = network.get_duplicate_groups()
    for i, group in enumerate(groups, 1):
        print(f"Group {i} ({len(group)} documents):")
        for doc in group:
            # Mark the representative (not in mapping)
            doc_idx = documents.index(doc)
            is_representative = doc_idx not in result["mapping"]
            marker = "👑" if is_representative else "  "
            print(f"  {marker} {doc}")
        print()

    # Show mapping details
    if result["mapping"]:
        print("🔄 DEDUPLICATION MAPPING:")
        print("-" * 40)
        for duplicate_idx, representative_idx in result["mapping"].items():
            duplicate_doc = documents[duplicate_idx]
            representative_doc = documents[representative_idx]
            print(f"'{duplicate_doc}'")
            print(f"  ↳ maps to: '{representative_doc}'")
            print()

    print("✅ Deduplication complete!")


if __name__ == "__main__":
    main()
