"""Test suite for SemanticNetwork class."""

import numpy as np
import pandas as pd
import pytest
import networkx as nx
from unittest.mock import Mock, patch

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
    def semantic_network(self, sample_docs):
        """SemanticNetwork instance with mock model for testing."""
        with patch("semnet.semnet.SentenceTransformer") as mock_st:
            # Mock the sentence transformer
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(len(sample_docs), 384)
            mock_st.return_value = mock_model

            network = SemanticNetwork(
                docs=sample_docs, embedding_model="test-model", verbose=False
            )
            return network

    def test_init_basic(self, sample_docs):
        """Test basic initialization."""
        with patch("semnet.semnet.SentenceTransformer"):
            network = SemanticNetwork(docs=sample_docs)

            assert network.docs == sample_docs
            assert network.weights is None
            assert network.n_trees == 10
            assert network.metric == "angular"
            assert network.verbose is False

    def test_init_with_weights(self, sample_docs, sample_weights):
        """Test initialization with weights."""
        with patch("semnet.semnet.SentenceTransformer"):
            network = SemanticNetwork(docs=sample_docs, weights=sample_weights)
            assert network.weights == sample_weights

    def test_init_weights_length_mismatch(self, sample_docs):
        """Test initialization fails with mismatched weights length."""
        with patch("semnet.semnet.SentenceTransformer"):
            with pytest.raises(
                ValueError, match="Weights length.*must match docs length"
            ):
                SemanticNetwork(docs=sample_docs, weights=[1.0, 2.0])

    def test_embed_documents(self, semantic_network, sample_docs):
        """Test document embedding generation."""
        embeddings = semantic_network.embed_documents()

        assert embeddings is not None
        assert embeddings.shape[0] == len(sample_docs)
        assert embeddings.shape[1] > 0
        assert semantic_network.embeddings is not None
        np.testing.assert_array_equal(embeddings, semantic_network.embeddings)

    def test_build_vector_index(self, semantic_network):
        """Test building vector index."""
        semantic_network.embed_documents()

        with patch("semnet.semnet.AnnoyIndex") as mock_annoy:
            mock_index = Mock()
            mock_annoy.return_value = mock_index

            index = semantic_network.build_vector_index()

            assert index is not None
            assert semantic_network.index is not None
            mock_index.build.assert_called_once_with(semantic_network.n_trees)

    def test_get_pairwise_similarities(self, semantic_network):
        """Test pairwise similarity computation."""
        semantic_network.embed_documents()

        with patch("semnet.semnet.AnnoyIndex") as mock_annoy:
            mock_index = Mock()
            mock_index.get_nns_by_item.return_value = ([0, 1], [0.0, 0.2])
            mock_annoy.return_value = mock_index
            semantic_network.build_vector_index()

            result = semantic_network.get_pairwise_similarities(
                thresh=0.7, inplace=False
            )

            assert isinstance(result, pd.DataFrame)
            assert semantic_network.neighbor_data is not None

    def test_build_graph(self, semantic_network, sample_docs):
        """Test graph building."""
        semantic_network.embed_documents()

        with patch("semnet.semnet.AnnoyIndex") as mock_annoy:
            mock_index = Mock()
            mock_index.get_nns_by_item.return_value = ([0, 1], [0.0, 0.2])
            mock_annoy.return_value = mock_index
            semantic_network.build_vector_index()

        semantic_network.get_pairwise_similarities(thresh=0.7)
        graph = semantic_network.build_graph()

        assert isinstance(graph, nx.Graph)
        assert semantic_network.graph is not None
        assert graph.number_of_nodes() == len(sample_docs)


@pytest.mark.slow
def test_real_model_integration():
    """Test with a real sentence transformer model (slower)."""
    docs = [
        "The cat sat on the mat",
        "A cat was sitting on a mat",
        "The dog ran quickly",
    ]

    # Use a small, fast model for testing
    network = SemanticNetwork(
        docs=docs, embedding_model="all-MiniLM-L6-v2", verbose=False
    )

    result = network.deduplicate_documents(thresh=0.8)

    # Basic sanity checks
    assert len(result["representatives"]) <= len(docs)
    assert result["stats"]["original_count"] == len(docs)
    assert isinstance(result["graph"], nx.Graph)
    assert isinstance(result["similarities"], pd.DataFrame)
