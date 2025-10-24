"""Test suite for SemanticNetwork class."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

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
            "Python is a programming language",
        ]

    @pytest.fixture
    def sample_embeddings(self, sample_docs):
        """Sample embeddings for testing."""
        np.random.seed(42)  # For reproducible tests
        return np.random.rand(len(sample_docs), 128)

    def test_init_basic(self):
        """Test basic initialization."""
        network = SemanticNetwork()

        assert network.n_trees == 10
        assert network.metric == "angular"
        assert network.verbose is False
        assert network.is_fitted_ is False

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        network = SemanticNetwork(metric="euclidean", thresh=0.5, verbose=True)
        assert network.metric == "euclidean"
        assert network.thresh == 0.5
        assert network.verbose is True

    def test_fit_basic(self, sample_docs, sample_embeddings):
        """Test basic fitting."""
        network = SemanticNetwork(verbose=False)

        # Test fit
        result = network.fit(sample_embeddings)
        assert result is network  # Should return self
        assert network.is_fitted_ is True
        assert network.embeddings_ is not None
        assert network.embeddings_.shape[0] == len(sample_docs)

    def test_transform_basic(self, sample_docs, sample_embeddings):
        """Test basic transform."""
        network = SemanticNetwork(verbose=False)
        network.fit(sample_embeddings)

        # Test transform
        graph = network.transform(labels=sample_docs)
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == len(sample_docs)

    def test_fit_transform(self, sample_docs, sample_embeddings):
        """Test fit_transform method."""
        network = SemanticNetwork(verbose=False)

        # Test fit_transform
        graph = network.fit_transform(sample_embeddings, labels=sample_docs)
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == len(sample_docs)
        assert network.is_fitted_ is True

    def test_transform_not_fitted(self):
        """Test transform fails when not fitted."""
        network = SemanticNetwork()
        with pytest.raises(ValueError, match="not fitted yet"):
            network.transform()

    def test_fit_with_custom_embeddings(self, sample_docs):
        """Test fitting with custom embeddings."""
        # Create custom embeddings
        custom_embeddings = np.random.rand(len(sample_docs), 256)

        network = SemanticNetwork(verbose=False)
        result = network.fit(custom_embeddings)

        assert result is network
        assert network.is_fitted_ is True
        assert network.embeddings_ is not None
        np.testing.assert_array_equal(network.embeddings_, custom_embeddings)
        assert network.embeddings_.shape == (len(sample_docs), 256)

    def test_fit_embeddings_shape_mismatch(self, sample_docs):
        """Test transform fails with mismatched embeddings and labels length."""
        # Create embeddings with wrong number of documents
        wrong_embeddings = np.random.rand(len(sample_docs) - 1, 128)

        network = SemanticNetwork()
        network.fit(wrong_embeddings)
        with pytest.raises(
            ValueError, match="Labels length.*must match embeddings length"
        ):
            network.transform(labels=sample_docs)  # This should fail

    def test_fit_with_node_data(self):
        """Test fitting with additional node data."""
        docs = ["doc1", "doc2", "doc3"]
        embeddings = np.random.rand(3, 128)
        node_data = {
            0: {"category": "cat1", "score": 0.8},
            1: {"category": "cat2", "score": 0.9},
            2: {"category": "cat1", "score": 0.7},
        }

        network = SemanticNetwork(verbose=False)
        network.fit(embeddings)
        graph = network.transform(labels=docs, node_data=node_data)

        # Check that node data was stored
        for node in graph.nodes():
            if node in node_data:
                for attr, value in node_data[node].items():
                    assert graph.nodes[node][attr] == value

    def test_fit_node_data_invalid_indices(self):
        """Test transform fails with invalid node indices in node_data."""
        docs = ["doc1", "doc2", "doc3"]
        embeddings = np.random.rand(3, 128)
        # Invalid index (3 is out of bounds for 3 documents)
        wrong_node_data = {3: {"category": "cat1"}}

        network = SemanticNetwork()
        network.fit(embeddings)
        with pytest.raises(
            ValueError,
            match="Node data contains invalid indices.*Indices must be",
        ):
            network.transform(labels=docs, node_data=wrong_node_data)

    def test_fit_node_data_single_values(self):
        """Test transform with single-value node data format."""
        docs = ["doc1", "doc2", "doc3"]
        embeddings = np.random.rand(3, 128)

        # Test that single values are accepted (converted to {'value': value})
        node_data_single = {0: "some_value", 1: 42}
        network = SemanticNetwork(verbose=False)
        network.fit(embeddings)
        graph = network.transform(labels=docs, node_data=node_data_single)

        assert graph.nodes[0]["value"] == "some_value"
        assert graph.nodes[1]["value"] == 42

    def test_fit_labels_length_mismatch(self):
        """Test transform fails with mismatched labels length."""
        embeddings = np.random.rand(3, 128)
        wrong_labels = ["doc1", "doc2"]  # Wrong length

        network = SemanticNetwork()
        network.fit(embeddings)
        with pytest.raises(
            ValueError, match="Labels length.*must match embeddings length"
        ):
            network.transform(labels=wrong_labels)

    def test_fit_with_defaults(self):
        """Test fitting with only embeddings (all other params default)."""
        embeddings = np.random.rand(3, 128)

        network = SemanticNetwork(verbose=False)
        network.fit(embeddings)
        graph = network.transform()

        assert network.is_fitted_ is True
        # Check that default labels are string indices
        assert network._labels == ["0", "1", "2"]
        # Check that default IDs are integer indices (0, 1, 2)
        assert graph.nodes[0]["id"] == 0
        assert graph.nodes[1]["id"] == 1
        assert graph.nodes[2]["id"] == 2
        # Check graph has correct node labels
        for i in range(3):
            assert graph.nodes[i]["label"] == str(i)

    def test_to_pandas_basic(self):
        """Test basic to_pandas functionality."""
        docs = ["doc1", "doc2", "doc3"]
        embeddings = np.random.rand(3, 128)

        network = SemanticNetwork(verbose=False)
        graph = network.fit_transform(embeddings, labels=docs)

        nodes, edges = network.to_pandas(graph)

        # Check nodes DataFrame
        assert isinstance(nodes, pd.DataFrame)
        assert len(nodes) == 3
        assert "label" in nodes.columns
        assert "id" in nodes.columns

        # Check that node labels match our docs
        node_labels = nodes["label"].tolist()
        assert set(node_labels) == set(docs)

        # Check edges DataFrame
        assert isinstance(edges, pd.DataFrame)

    def test_to_pandas_with_node_data(self):
        """Test to_pandas with custom node data."""
        docs = ["doc1", "doc2", "doc3"]
        embeddings = np.random.rand(3, 128)
        node_data = {
            0: {"category": "tech", "priority": 1},
            1: {"category": "science", "priority": 2},
            2: {"category": "tech", "priority": 1},
        }

        network = SemanticNetwork(verbose=False)
        graph = network.fit_transform(
            embeddings, labels=docs, node_data=node_data
        )

        nodes, edges = network.to_pandas(graph)

        # Check that custom node data is included
        assert "category" in nodes.columns
        assert "priority" in nodes.columns

        # Check specific values
        assert nodes.loc[0, "category"] == "tech"
        assert nodes.loc[1, "priority"] == 2
        assert nodes.loc[2, "category"] == "tech"

    def test_to_pandas_with_similarities(self):
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
        embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )

        network = SemanticNetwork(verbose=False, thresh=0.8)
        graph = network.fit_transform(embeddings, labels=docs)

        nodes, edges = network.to_pandas(graph)

        # Should have edges due to high similarity
        if len(edges) > 0:
            assert "source" in edges.columns
            assert "target" in edges.columns
            assert "weight" in edges.columns

            # Check that weights are above threshold
            assert (edges["weight"] >= 0.8).all()

    def test_to_pandas_no_graph_provided(self):
        """Test to_pandas fails when no graph provided."""
        network = SemanticNetwork()

        with pytest.raises(ValueError, match="No graph provided"):
            network.to_pandas()

    def test_transform_with_custom_thresholds(
        self, sample_docs, sample_embeddings
    ):
        """Test transform with custom threshold and top_k overrides."""
        network = SemanticNetwork(verbose=False, thresh=0.9, top_k=50)
        network.fit(sample_embeddings)

        # Transform with different thresholds
        graph1 = network.transform(
            thresh=0.1, top_k=10, labels=sample_docs
        )  # Lower threshold = more edges
        graph2 = network.transform(
            thresh=0.95, top_k=100, labels=sample_docs
        )  # Higher threshold = fewer edges

        # Both should be valid graphs
        assert isinstance(graph1, nx.Graph)
        assert isinstance(graph2, nx.Graph)
        assert graph1.number_of_nodes() == len(sample_docs)
        assert graph2.number_of_nodes() == len(sample_docs)


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

    # Use semnet to build graph
    network = SemanticNetwork(verbose=False, thresh=0.8)
    graph = network.fit_transform(embeddings, labels=docs)

    # Basic sanity checks
    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == len(docs)
    assert network.is_fitted_ is True
