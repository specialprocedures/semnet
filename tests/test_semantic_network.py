"""Test suite for SemanticNetwork class."""

import numpy as np
import pandas as pd
import pytest
import networkx as nx

from semnet import SemanticNetwork


class TestSemanticNetwork:
    """Test cases for SemanticNetwork class."""

    @pytest.fixture
    def sample_docs(self):
        """Sample documents for testing."""
        return [
            "The cat sat on the mat",
            "A cat was sitting on a mat",
            "The dog ran in the park",
            "A dog was running in the park",
            "Python is a programming language",
            "Machine learning with Python",
        ]

    @pytest.fixture
    def sample_weights(self):
        """Sample weights corresponding to sample_docs."""
        return [1.0, 0.5, 2.0, 1.5, 3.0, 2.5]

    @pytest.fixture
    def sample_embeddings(self, sample_docs):
        """Sample embeddings for testing."""
        np.random.seed(42)  # For reproducible tests
        return np.random.rand(len(sample_docs), 128)

    def test_init_basic(self, sample_docs):
        """Test basic initialization."""
        network = SemanticNetwork()

        assert network.n_trees == 10
        assert network.metric == "angular"
        assert network.verbose is False
        assert network.is_fitted_ is False

    def test_init_with_weights(self, sample_docs, sample_weights):
        """Test initialization with custom parameters."""
        network = SemanticNetwork(metric="euclidean", thresh=0.5, verbose=True)
        assert network.metric == "euclidean"
        assert network.thresh == 0.5
        assert network.verbose is True

    def test_fit_weights_length_mismatch(self, sample_docs, sample_embeddings):
        """Test fit fails with mismatched weights length."""
        network = SemanticNetwork()
        with pytest.raises(
            ValueError, match="Weights length.*must match embeddings length"
        ):
            network.fit(sample_embeddings, labels=sample_docs, weights=[1.0, 2.0])

    def test_fit_and_transform(self, sample_docs, sample_embeddings):
        """Test fitting and transforming."""
        network = SemanticNetwork(verbose=False)

        # Test fit
        result = network.fit(sample_embeddings, labels=sample_docs)
        assert result is network  # Should return self
        assert network.is_fitted_ is True
        assert network.embeddings_ is not None
        assert network.embeddings_.shape[0] == len(sample_docs)

        # Test transform
        representatives = network.transform()
        assert isinstance(representatives, list)
        assert len(representatives) <= len(sample_docs)

    def test_fit_transform(self, sample_docs, sample_embeddings):
        """Test fit_transform method."""
        network = SemanticNetwork(verbose=False)

        # Test fit_transform
        representatives = network.fit_transform(sample_embeddings, labels=sample_docs)
        assert isinstance(representatives, list)
        assert len(representatives) <= len(sample_docs)
        assert network.is_fitted_ is True

    def test_transform_not_fitted(self):
        """Test transform fails when not fitted."""
        network = SemanticNetwork()
        with pytest.raises(ValueError, match="not fitted yet"):
            network.transform()

    def test_get_duplicate_groups(self, sample_docs, sample_embeddings):
        """Test getting duplicate groups."""
        network = SemanticNetwork(verbose=False)
        network.fit(sample_embeddings, labels=sample_docs)

        groups = network.get_duplicate_groups()
        assert isinstance(groups, list)

    def test_fit_with_custom_embeddings(self, sample_docs):
        """Test fitting with custom embeddings."""
        # Create custom embeddings
        custom_embeddings = np.random.rand(len(sample_docs), 128)

        network = SemanticNetwork(verbose=False)

        result = network.fit(custom_embeddings, labels=sample_docs)

        assert result is network
        assert network.is_fitted_ is True
        assert network.embeddings_ is not None
        np.testing.assert_array_equal(network.embeddings_, custom_embeddings)
        assert network.embeddings_.shape == (len(sample_docs), 128)

    def test_fit_transform_with_custom_embeddings(self, sample_docs):
        """Test fit_transform with custom embeddings."""
        # Create custom embeddings
        custom_embeddings = np.random.rand(len(sample_docs), 256)

        network = SemanticNetwork(verbose=False)

        # Test fit_transform with custom embeddings
        representatives = network.fit_transform(custom_embeddings, labels=sample_docs)

        assert isinstance(representatives, list)
        assert len(representatives) <= len(sample_docs)
        assert network.is_fitted_ is True
        np.testing.assert_array_equal(network.embeddings_, custom_embeddings)

    def test_fit_embeddings_shape_mismatch(self, sample_docs):
        """Test fit fails with mismatched embeddings shape."""
        # Create embeddings with wrong number of documents
        wrong_embeddings = np.random.rand(len(sample_docs) - 1, 128)

        network = SemanticNetwork()
        with pytest.raises(
            ValueError, match="Labels length.*must match embeddings length"
        ):
            network.fit(wrong_embeddings, labels=sample_docs)

    def test_fit_with_blocks_1d(self, sample_docs):
        """Test fitting with 1D blocks."""
        # Create blocks - group similar documents together
        blocks = ["A", "A", "B", "B", "C", "C"]  # Same length as sample_docs
        custom_embeddings = np.random.rand(len(sample_docs), 128)

        network = SemanticNetwork(verbose=False, thresh=0.5)

        result = network.fit(custom_embeddings, labels=sample_docs, blocks=blocks)

        assert result is network
        assert network.is_fitted_ is True
        assert network.blocks_ is not None
        assert network.blocks_.shape == (len(sample_docs), 1)
        np.testing.assert_array_equal(network.blocks_.flatten(), blocks)

    def test_fit_with_blocks_2d(self, sample_docs):
        """Test fitting with 2D blocks (multiple blocking variables)."""
        # Create 2D blocks - company and department
        blocks = [
            ["CompanyA", "Dept1"],
            ["CompanyA", "Dept1"],
            ["CompanyA", "Dept2"],
            ["CompanyB", "Dept1"],
            ["CompanyB", "Dept2"],
            ["CompanyC", "Dept1"],
        ]
        custom_embeddings = np.random.rand(len(sample_docs), 128)

        network = SemanticNetwork(verbose=False, thresh=0.5)

        result = network.fit(custom_embeddings, labels=sample_docs, blocks=blocks)

        assert result is network
        assert network.is_fitted_ is True
        assert network.blocks_ is not None
        assert network.blocks_.shape == (len(sample_docs), 2)

    def test_fit_blocks_length_mismatch(self, sample_docs, sample_embeddings):
        """Test fit fails with mismatched blocks length."""
        wrong_blocks = ["A", "B"]  # Wrong length

        network = SemanticNetwork()
        with pytest.raises(
            ValueError, match="Blocks length.*must match embeddings length"
        ):
            network.fit(sample_embeddings, labels=sample_docs, blocks=wrong_blocks)

    def test_fit_transform_with_blocks(self, sample_docs):
        """Test fit_transform with blocks."""
        blocks = ["Group1", "Group1", "Group2", "Group2", "Group3", "Group3"]
        custom_embeddings = np.random.rand(len(sample_docs), 128)

        network = SemanticNetwork(verbose=False, thresh=0.5)

        representatives = network.fit_transform(
            custom_embeddings, labels=sample_docs, blocks=blocks
        )

        assert isinstance(representatives, list)
        assert len(representatives) <= len(sample_docs)
        assert network.is_fitted_ is True
        assert network.blocks_ is not None

    def test_blocking_reduces_comparisons(self, sample_docs):
        """Test that blocking reduces the number of similarity comparisons."""
        # Create embeddings where all documents are very similar
        base_embedding = np.random.rand(128)
        similar_embeddings = np.array(
            [
                base_embedding + 0.01 * np.random.rand(128)
                for _ in range(len(sample_docs))
            ]
        )

        # Normalize embeddings
        similar_embeddings = similar_embeddings / np.linalg.norm(
            similar_embeddings, axis=1, keepdims=True
        )

        # Test without blocks
        network_no_blocks = SemanticNetwork(verbose=False, thresh=0.8)
        network_no_blocks.fit(similar_embeddings, labels=sample_docs)
        stats_no_blocks = network_no_blocks.get_deduplication_stats()

        # Test with blocks that separate documents
        blocks = [
            f"Block{i}" for i in range(len(sample_docs))
        ]  # Each doc in its own block
        network_with_blocks = SemanticNetwork(verbose=False, thresh=0.8)
        network_with_blocks.fit(similar_embeddings, labels=sample_docs, blocks=blocks)
        stats_with_blocks = network_with_blocks.get_deduplication_stats()

        # With blocks, there should be no similarities found (each doc in separate block)
        assert stats_with_blocks["similarity_pairs"] == 0
        # Without blocks, there should be many similarities (all docs are similar)
        assert stats_no_blocks["similarity_pairs"] > 0


def test_fit_with_custom_ids():
    """Test fitting with custom IDs."""
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.random.rand(3, 128)
    custom_ids = ["id_a", "id_b", "id_c"]

    network = SemanticNetwork(verbose=False)
    network.fit(embeddings, labels=docs, ids=custom_ids)

    assert network.is_fitted_ is True
    # Check that IDs are stored in graph nodes
    for i, node_id in enumerate(custom_ids):
        assert network.graph_.nodes[i]["id"] == node_id


def test_fit_with_node_data():
    """Test fitting with additional node data."""
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.random.rand(3, 128)
    node_data = {
        0: {"category": "cat1", "score": 0.8},
        1: {"category": "cat2", "score": 0.9},
        2: {"category": "cat1", "score": 0.7},
    }

    network = SemanticNetwork(verbose=False)
    network.fit(embeddings, labels=docs, node_data=node_data)

    assert network.graph_ is not None
    # Check that node data was stored
    for node in network.graph_.nodes():
        if node in node_data:
            for attr, value in node_data[node].items():
                assert network.graph_.nodes[node][attr] == value


def test_fit_node_data_invalid_indices():
    """Test fit fails with invalid node indices in node_data."""
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.random.rand(3, 128)
    # Invalid index (3 is out of bounds for 3 documents)
    wrong_node_data = {3: {"category": "cat1"}}

    network = SemanticNetwork()
    with pytest.raises(
        ValueError, match="Node data contains invalid indices.*Indices must be"
    ):
        network.fit(embeddings, labels=docs, node_data=wrong_node_data)


def test_fit_node_data_invalid_format():
    """Test fit fails with invalid node_data format."""
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.random.rand(3, 128)

    # Test string keys (old format)
    wrong_node_data1 = {"category": ["cat1", "cat2", "cat1"]}
    network = SemanticNetwork()
    with pytest.raises(ValueError, match="Node data keys must be integer node indices"):
        network.fit(embeddings, labels=docs, node_data=wrong_node_data1)

    # Test that single values are accepted (converted to {'value': value})
    node_data_single = {0: "some_value", 1: 42}
    network.fit(embeddings, labels=docs, node_data=node_data_single)
    assert network.graph_.nodes[0]["value"] == "some_value"
    assert network.graph_.nodes[1]["value"] == 42


def test_fit_labels_length_mismatch():
    """Test fit fails with mismatched labels length."""
    embeddings = np.random.rand(3, 128)
    wrong_labels = ["doc1", "doc2"]  # Wrong length

    network = SemanticNetwork()
    with pytest.raises(ValueError, match="Labels length.*must match embeddings length"):
        network.fit(embeddings, labels=wrong_labels)


def test_fit_ids_length_mismatch():
    """Test fit fails with mismatched IDs length."""
    embeddings = np.random.rand(3, 128)
    wrong_ids = ["id1", "id2"]  # Wrong length

    network = SemanticNetwork()
    with pytest.raises(ValueError, match="IDs length.*must match embeddings length"):
        network.fit(embeddings, ids=wrong_ids)


def test_fit_with_defaults():
    """Test fitting with only embeddings (all other params default)."""
    embeddings = np.random.rand(3, 128)

    network = SemanticNetwork(verbose=False)
    network.fit(embeddings)

    assert network.is_fitted_ is True
    # Check that default labels are string indices
    assert network._labels == ["0", "1", "2"]
    # Check that default IDs are integer indices
    assert network._ids == [0, 1, 2]
    # Check graph has correct node names
    for i in range(3):
        assert network.graph_.nodes[i]["name"] == str(i)


def test_to_pandas_basic():
    """Test basic to_pandas functionality."""
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.random.rand(3, 128)

    network = SemanticNetwork(
        verbose=False, thresh=0.0
    )  # Low threshold to ensure connections
    network.fit(embeddings, labels=docs)

    nodes, edges = network.to_pandas()

    # Check nodes DataFrame
    assert isinstance(nodes, pd.DataFrame)
    assert len(nodes) == 3
    assert "name" in nodes.columns
    assert "weight" in nodes.columns
    assert "id" in nodes.columns

    # Check that node names match our docs
    node_names = nodes["name"].tolist()
    assert set(node_names) == set(docs)

    # Check edges DataFrame
    assert isinstance(edges, pd.DataFrame)
    # Edges may or may not exist depending on similarity, but should be a DataFrame


def test_to_pandas_with_node_data():
    """Test to_pandas with custom node data."""
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.random.rand(3, 128)
    node_data = {
        0: {"category": "tech", "priority": 1},
        1: {"category": "science", "priority": 2},
        2: {"category": "tech", "priority": 1},
    }

    network = SemanticNetwork(verbose=False)
    network.fit(embeddings, labels=docs, node_data=node_data)

    nodes, edges = network.to_pandas()

    # Check that custom node data is included
    assert "category" in nodes.columns
    assert "priority" in nodes.columns

    # Check specific values
    assert nodes.loc[0, "category"] == "tech"
    assert nodes.loc[1, "priority"] == 2
    assert nodes.loc[2, "category"] == "tech"


def test_to_pandas_with_custom_ids():
    """Test to_pandas with custom IDs."""
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.random.rand(3, 128)
    custom_ids = ["id_a", "id_b", "id_c"]

    network = SemanticNetwork(verbose=False)
    network.fit(embeddings, labels=docs, ids=custom_ids)

    nodes, edges = network.to_pandas()

    # Check that custom IDs are included
    assert "id" in nodes.columns
    assert nodes.loc[0, "id"] == "id_a"
    assert nodes.loc[1, "id"] == "id_b"
    assert nodes.loc[2, "id"] == "id_c"


def test_to_pandas_with_weights():
    """Test to_pandas with custom weights."""
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.random.rand(3, 128)
    weights = [1.5, 2.0, 0.5]

    network = SemanticNetwork(verbose=False)
    network.fit(embeddings, labels=docs, weights=weights)

    nodes, edges = network.to_pandas()

    # Check that weights are included
    assert "weight" in nodes.columns
    assert nodes.loc[0, "weight"] == 1.5
    assert nodes.loc[1, "weight"] == 2.0
    assert nodes.loc[2, "weight"] == 0.5


def test_to_pandas_with_similarities():
    """Test to_pandas with forced similarities."""
    docs = ["doc1", "doc2", "doc3"]
    # Create very similar embeddings to ensure connections
    base_embedding = np.random.rand(128)
    embeddings = np.array(
        [
            base_embedding + 0.001 * np.random.rand(128),
            base_embedding + 0.001 * np.random.rand(128),
            base_embedding + 0.001 * np.random.rand(128),
        ]
    )

    # Normalize embeddings for consistent similarity calculation
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    network = SemanticNetwork(verbose=False, thresh=0.8)
    network.fit(embeddings, labels=docs)

    nodes, edges = network.to_pandas()

    # Should have edges due to high similarity
    if len(edges) > 0:
        assert "source" in edges.columns
        assert "target" in edges.columns
        assert "similarity" in edges.columns

        # Check that similarities are above threshold
        assert (edges["similarity"] >= 0.8).all()


def test_to_pandas_no_edges():
    """Test to_pandas when no similarities are found."""
    docs = ["doc1", "doc2", "doc3"]
    # Create very different embeddings
    embeddings = np.array(
        [
            [1.0] + [0.0] * 127,  # Different directions
            [0.0] + [1.0] + [0.0] * 126,
            [0.0] * 2 + [1.0] + [0.0] * 125,
        ]
    )

    network = SemanticNetwork(verbose=False, thresh=0.9)  # High threshold
    network.fit(embeddings, labels=docs)

    nodes, edges = network.to_pandas()

    # Should still have nodes
    assert len(nodes) == 3
    assert "name" in nodes.columns

    # Edges may be empty but should be a DataFrame
    assert isinstance(edges, pd.DataFrame)


def test_to_pandas_not_fitted():
    """Test to_pandas fails when not fitted."""
    network = SemanticNetwork()

    with pytest.raises(ValueError, match="not fitted yet"):
        network.to_pandas()


def test_to_pandas_single_value_node_data():
    """Test to_pandas with single-value node data format."""
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.random.rand(3, 128)
    # Use the single-value format that gets converted to {'value': value}
    node_data = {0: "author1", 1: "author2", 2: "author3"}

    network = SemanticNetwork(verbose=False)
    network.fit(embeddings, labels=docs, node_data=node_data)

    nodes, edges = network.to_pandas()

    # Check that single values are stored under 'value' column
    assert "value" in nodes.columns
    assert nodes.loc[0, "value"] == "author1"
    assert nodes.loc[1, "value"] == "author2"
    assert nodes.loc[2, "value"] == "author3"


def test_to_pandas_with_custom_graph():
    """Test to_pandas with a custom graph parameter."""
    docs = ["doc1", "doc2", "doc3", "doc4"]
    embeddings = np.random.rand(4, 128)

    network = SemanticNetwork(verbose=False)
    network.fit(embeddings, labels=docs)

    # Create a subgraph with only nodes 0, 1, 2
    subgraph = network.graph_.subgraph([0, 1, 2])

    # Export the subgraph
    nodes, edges = network.to_pandas(subgraph)

    # Should only have 3 nodes
    assert len(nodes) == 3
    assert set(nodes.index) == {0, 1, 2}

    # Check that node attributes are preserved
    assert "name" in nodes.columns
    assert nodes.loc[0, "name"] == "doc1"
    assert nodes.loc[1, "name"] == "doc2"
    assert nodes.loc[2, "name"] == "doc3"


def test_to_pandas_with_empty_custom_graph():
    """Test to_pandas with an empty custom graph."""
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.random.rand(3, 128)

    network = SemanticNetwork(verbose=False)
    network.fit(embeddings, labels=docs)

    # Create an empty graph
    empty_graph = nx.Graph()

    # Export the empty graph
    nodes, edges = network.to_pandas(empty_graph)

    # Should have empty DataFrames
    assert len(nodes) == 0
    assert len(edges) == 0
    assert isinstance(nodes, pd.DataFrame)
    assert isinstance(edges, pd.DataFrame)


def test_to_pandas_custom_graph_with_attributes():
    """Test to_pandas with a custom graph that has custom attributes."""
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.random.rand(3, 128)

    network = SemanticNetwork(verbose=False)
    network.fit(embeddings, labels=docs)

    # Create a custom graph with additional attributes
    custom_graph = nx.Graph()
    custom_graph.add_node(0, name="Custom Node 1", custom_attr="value1")
    custom_graph.add_node(1, name="Custom Node 2", custom_attr="value2")
    custom_graph.add_edge(0, 1, weight=0.5, custom_edge_attr="edge_value")

    # Export the custom graph
    nodes, edges = network.to_pandas(custom_graph)

    # Check nodes
    assert len(nodes) == 2
    assert "custom_attr" in nodes.columns
    assert nodes.loc[0, "custom_attr"] == "value1"
    assert nodes.loc[1, "custom_attr"] == "value2"

    # Check edges
    assert len(edges) == 1
    assert "custom_edge_attr" in edges.columns
    assert edges.iloc[0]["custom_edge_attr"] == "edge_value"


def test_to_pandas_unfitted_with_custom_graph():
    """Test to_pandas with custom graph on unfitted network."""
    # Create unfitted network
    network = SemanticNetwork()

    # Create a custom graph
    custom_graph = nx.Graph()
    custom_graph.add_node(0, name="Node 1")
    custom_graph.add_node(1, name="Node 2")
    custom_graph.add_edge(0, 1, similarity=0.8)

    # Should work even though network is not fitted
    nodes, edges = network.to_pandas(custom_graph)

    assert len(nodes) == 2
    assert len(edges) == 1
    assert nodes.loc[0, "name"] == "Node 1"
    assert edges.iloc[0]["similarity"] == 0.8


def test_to_pandas_unfitted_no_graph():
    """Test to_pandas fails when unfitted and no graph provided."""
    network = SemanticNetwork()

    with pytest.raises(ValueError, match="not fitted yet.*provide a graph parameter"):
        network.to_pandas()


@pytest.mark.slow
def test_real_model_integration():
    """Test with real embeddings from sentence-transformers (slower)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not available for integration test")

    docs = [
        "The cat sat on the mat",
        "A cat was sitting on a mat",
        "The dog ran quickly",
    ]

    # Generate embeddings using sentence-transformers
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(docs)

    # Use semnet without embedding generation
    network = SemanticNetwork(verbose=False, thresh=0.8)
    representatives = network.fit_transform(embeddings, labels=docs)

    # Basic sanity checks
    assert len(representatives) <= len(docs)
    assert network.is_fitted_ is True
    stats = network.get_deduplication_stats()
    assert stats["original_count"] == len(docs)
    assert isinstance(stats, dict)
