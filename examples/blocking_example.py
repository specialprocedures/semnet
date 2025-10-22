#!/usr/bin/env python3
"""
Example of using Semnet with blocking for improved performance.

This example shows how to use blocking to only compare documents
within the same block(s), which can dramatically improve performance
for large datasets where you know certain groupings make sense.

Users must provide their own embeddings.
"""

import numpy as np
from semnet import SemanticNetwork


def main():
    """Run blocking example."""

    # Example: Employee names with their companies and departments
    documents = [
        "John Smith, Software Engineer",
        "J. Smith, Developer",
        "John S., Programmer",
        "Jane Doe, Product Manager",
        "J. Doe, PM",
        "Mike Johnson, Sales Rep",
        "Michael Johnson, Account Executive",
        "Sarah Wilson, Marketing Manager",
        "S. Wilson, Marketing Lead",
        "Tom Brown, HR Director",
        "Thomas Brown, Human Resources",
        "Lisa Davis, Finance Analyst",
    ]

    # Blocking variables: company and department
    # In practice, these might come from structured data
    companies = [
        "TechCorp",
        "TechCorp",
        "TechCorp",  # John Smith variations
        "TechCorp",
        "TechCorp",  # Jane Doe variations
        "SalesCo",
        "SalesCo",  # Mike Johnson variations
        "MarketInc",
        "MarketInc",  # Sarah Wilson variations
        "TechCorp",
        "TechCorp",  # Tom Brown variations
        "FinanceGroup",  # Lisa Davis (single entry)
    ]

    departments = [
        "Engineering",
        "Engineering",
        "Engineering",  # John Smith variations
        "Product",
        "Product",  # Jane Doe variations
        "Sales",
        "Sales",  # Mike Johnson variations
        "Marketing",
        "Marketing",  # Sarah Wilson variations
        "HR",
        "HR",  # Tom Brown variations
        "Finance",  # Lisa Davis (single entry)
    ]

    print("Blocking Example for Employee Deduplication")
    print("=" * 50)
    print(f"Input: {len(documents)} employee records")
    print()

    # Create blocks using both company and department
    # This means only employees in the same company AND department will be compared
    blocks = list(zip(companies, departments))

    print("Block structure (Company, Department):")
    unique_blocks = sorted(set(blocks))
    for i, block in enumerate(unique_blocks):
        block_docs = [doc for j, doc in enumerate(documents) if blocks[j] == block]
        print(f"  {block}: {len(block_docs)} employees")
    print()

    # Create embeddings that make similar names actually similar
    np.random.seed(42)
    base_embeddings = np.random.rand(len(documents), 128)

    # Make similar names more similar within their groups
    similar_pairs = [
        (0, 1, 2),  # John Smith variations
        (3, 4),  # Jane Doe variations
        (5, 6),  # Mike Johnson variations
        (7, 8),  # Sarah Wilson variations
        (9, 10),  # Tom Brown variations
    ]

    for group in similar_pairs:
        # Average the embeddings to make them similar
        group_mean = np.mean([base_embeddings[i] for i in group], axis=0)
        for i in group:
            base_embeddings[i] = 0.7 * group_mean + 0.3 * base_embeddings[i]

    # Normalize embeddings
    embeddings = base_embeddings / np.linalg.norm(
        base_embeddings, axis=1, keepdims=True
    )

    print("Testing WITHOUT blocking:")
    print("-" * 30)

    # Test without blocking first
    network_no_blocks = SemanticNetwork(
        thresh=0.3,
        verbose=True,
    )

    representatives_no_blocks = network_no_blocks.fit_transform(
        embeddings, labels=documents
    )

    stats_no_blocks = network_no_blocks.get_deduplication_stats()
    print(f"Representatives found: {len(representatives_no_blocks)}")
    print(f"Similarity pairs: {stats_no_blocks['similarity_pairs']}")
    print(f"Reduction ratio: {stats_no_blocks['reduction_ratio']:.1%}")
    print()

    print("Testing WITH blocking:")
    print("-" * 30)

    # Test with blocking
    network_with_blocks = SemanticNetwork(
        thresh=0.3,
        verbose=True,
    )

    representatives_with_blocks = network_with_blocks.fit_transform(
        embeddings, labels=documents, blocks=blocks
    )

    stats_with_blocks = network_with_blocks.get_deduplication_stats()
    print(f"Representatives found: {len(representatives_with_blocks)}")
    print(f"Similarity pairs: {stats_with_blocks['similarity_pairs']}")
    print(f"Reduction ratio: {stats_with_blocks['reduction_ratio']:.1%}")
    print()

    print("RESULTS WITH BLOCKING:")
    print("=" * 40)
    for i, rep in enumerate(representatives_with_blocks, 1):
        print(f"{i:2d}. {rep}")
    print()

    # Show duplicate groups
    groups = network_with_blocks.get_duplicate_groups()
    if groups:
        print("DUPLICATE GROUPS (within same company & department):")
        print("-" * 50)
        mapping = network_with_blocks.transform(return_representatives=False)
        for i, group in enumerate(groups, 1):
            print(f"Group {i} ({len(group)} employees):")
            for doc in group:
                doc_idx = documents.index(doc)
                is_representative = (
                    doc_idx not in mapping if isinstance(mapping, dict) else True
                )
                company, dept = blocks[doc_idx]
                marker = "*" if is_representative else " "
                print(f"  {marker} {doc} [{company}, {dept}]")
            print()

    print(f"Performance benefit: Blocking reduced similarity comparisons from")
    print(
        f"  {stats_no_blocks['similarity_pairs']} to {stats_with_blocks['similarity_pairs']}"
    )
    print(
        f"  ({100 * (1 - stats_with_blocks['similarity_pairs'] / max(1, stats_no_blocks['similarity_pairs'])):.1f}% reduction)"
    )


if __name__ == "__main__":
    main()
