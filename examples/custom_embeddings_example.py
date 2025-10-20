#!/usr/bin/env python3
"""
Example of using Semnet with custom embeddings.

This example shows how to use pre-computed embeddings instead of
generating them with sentence transformers.
"""

import numpy as np
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
    base_embeddings[1] = 0.7 * base_embeddings[0] + 0.3 * base_embeddings[1]

    # Documents 2 and 3 (dog documents)
    base_embeddings[3] = 0.7 * base_embeddings[2] + 0.3 * base_embeddings[3]

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

    # Fit using custom embeddings
    representatives = network.fit_transform(
        documents, embeddings=custom_embeddings, return_representatives=True
    )

    print()
    print("DEDUPLICATION RESULTS")
    print("=" * 30)

    # Print statistics
    stats = network.get_deduplication_stats()
    print(f"Original documents: {stats['original_count']}")
    print(f"After deduplication: {stats['deduplicated_count']}")
    print(f"Duplicates removed: {stats['duplicates_found']}")
    print(f"Reduction ratio: {stats['reduction_ratio']:.1%}")
    print()

    # Show representative documents
    print("REPRESENTATIVE DOCUMENTS:")
    print("-" * 30)
    for i, doc in enumerate(representatives, 1):
        print(f"{i:2d}. {doc}")
    print()

    # Show duplicate groups
    groups = network.get_duplicate_groups()
    if groups:
        print("DUPLICATE GROUPS:")
        print("-" * 30)
        mapping_dict = network.transform(return_representatives=False)
        for i, group in enumerate(groups, 1):
            print(f"Group {i} ({len(group)} documents):")
            for doc in group:
                doc_idx = documents.index(doc)
                is_representative = (
                    doc_idx not in mapping_dict
                    if isinstance(mapping_dict, dict)
                    else True
                )
                marker = "*" if is_representative else " "
                print(f"  {marker} {doc}")
            print()
    else:
        print("No duplicate groups found with the current threshold.")

    print("Custom embeddings example complete!")


if __name__ == "__main__":
    main()
