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

            network = SemanticNetwork(embedding_model="test-model", verbose=False)
            # Fit the network with sample docs
            network.fit(sample_docs)
            return network

    def test_init_basic(self, sample_docs):
        """Test basic initialization."""
        network = SemanticNetwork()

        assert network.embedding_model_name == "BAAI/bge-base-en-v1.5"
        assert network.n_trees == 10
        assert network.metric == "angular"
        assert network.verbose is False
        assert network.is_fitted_ is False

    def test_init_with_weights(self, sample_docs, sample_weights):
        """Test initialization with custom parameters."""
        network = SemanticNetwork(
            embedding_model="custom-model", metric="euclidean", thresh=0.5, verbose=True
        )
        assert network.embedding_model_name == "custom-model"
        assert network.metric == "euclidean"
        assert network.thresh == 0.5
        assert network.verbose is True

    def test_fit_weights_length_mismatch(self, sample_docs):
        """Test fit fails with mismatched weights length."""
        network = SemanticNetwork()
        with pytest.raises(ValueError, match="Weights length.*must match X length"):
            network.fit(sample_docs, weights=[1.0, 2.0])

    def test_fit_and_transform(self, sample_docs):
        """Test fitting and transforming."""
        with patch("semnet.semnet.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(len(sample_docs), 384)
            mock_st.return_value = mock_model

            network = SemanticNetwork(verbose=False)

            # Test fit
            result = network.fit(sample_docs)
            assert result is network  # Should return self
            assert network.is_fitted_ is True
            assert network.embeddings_ is not None
            assert network.embeddings_.shape[0] == len(sample_docs)

            # Test transform
            representatives = network.transform()
            assert isinstance(representatives, list)
            assert len(representatives) <= len(sample_docs)

    def test_fit_transform(self, sample_docs):
        """Test fit_transform method."""
        with patch("semnet.semnet.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(len(sample_docs), 384)
            mock_st.return_value = mock_model

            network = SemanticNetwork(verbose=False)

            # Test fit_transform
            representatives = network.fit_transform(sample_docs)
            assert isinstance(representatives, list)
            assert len(representatives) <= len(sample_docs)
            assert network.is_fitted_ is True

    def test_transform_not_fitted(self):
        """Test transform fails when not fitted."""
        network = SemanticNetwork()
        with pytest.raises(ValueError, match="not fitted yet"):
            network.transform()

    def test_get_duplicate_groups(self, sample_docs):
        """Test getting duplicate groups."""
        with patch("semnet.semnet.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(len(sample_docs), 384)
            mock_st.return_value = mock_model

            network = SemanticNetwork(verbose=False)
            network.fit(sample_docs)

            groups = network.get_duplicate_groups()
            assert isinstance(groups, list)

    def test_fit_with_custom_embeddings(self, sample_docs):
        """Test fitting with custom embeddings."""
        # Create custom embeddings
        custom_embeddings = np.random.rand(len(sample_docs), 128)

        network = SemanticNetwork(verbose=False)

        # Should not call SentenceTransformer when custom embeddings provided
        result = network.fit(sample_docs, embeddings=custom_embeddings)

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
        representatives = network.fit_transform(
            sample_docs, embeddings=custom_embeddings
        )

        assert isinstance(representatives, list)
        assert len(representatives) <= len(sample_docs)
        assert network.is_fitted_ is True
        np.testing.assert_array_equal(network.embeddings_, custom_embeddings)

    def test_fit_embeddings_shape_mismatch(self, sample_docs):
        """Test fit fails with mismatched embeddings shape."""
        # Create embeddings with wrong number of documents
        wrong_embeddings = np.random.rand(len(sample_docs) - 1, 128)

        network = SemanticNetwork()
        with pytest.raises(ValueError, match="Embeddings shape.*must match X length"):
            network.fit(sample_docs, embeddings=wrong_embeddings)

    def test_fit_with_blocks_1d(self, sample_docs):
        """Test fitting with 1D blocks."""
        # Create blocks - group similar documents together
        blocks = ["A", "A", "B", "B", "C", "C"]  # Same length as sample_docs
        custom_embeddings = np.random.rand(len(sample_docs), 128)

        network = SemanticNetwork(verbose=False, thresh=0.5)

        result = network.fit(sample_docs, embeddings=custom_embeddings, blocks=blocks)

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

        result = network.fit(sample_docs, embeddings=custom_embeddings, blocks=blocks)

        assert result is network
        assert network.is_fitted_ is True
        assert network.blocks_ is not None
        assert network.blocks_.shape == (len(sample_docs), 2)

    def test_fit_blocks_length_mismatch(self, sample_docs):
        """Test fit fails with mismatched blocks length."""
        wrong_blocks = ["A", "B"]  # Wrong length

        network = SemanticNetwork()
        with pytest.raises(ValueError, match="Blocks length.*must match X length"):
            network.fit(sample_docs, blocks=wrong_blocks)

    def test_fit_transform_with_blocks(self, sample_docs):
        """Test fit_transform with blocks."""
        blocks = ["Group1", "Group1", "Group2", "Group2", "Group3", "Group3"]
        custom_embeddings = np.random.rand(len(sample_docs), 128)

        network = SemanticNetwork(verbose=False, thresh=0.5)

        representatives = network.fit_transform(
            sample_docs, embeddings=custom_embeddings, blocks=blocks
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
        network_no_blocks.fit(sample_docs, embeddings=similar_embeddings)
        stats_no_blocks = network_no_blocks.get_deduplication_stats()

        # Test with blocks that separate documents
        blocks = [
            f"Block{i}" for i in range(len(sample_docs))
        ]  # Each doc in its own block
        network_with_blocks = SemanticNetwork(verbose=False, thresh=0.8)
        network_with_blocks.fit(
            sample_docs, embeddings=similar_embeddings, blocks=blocks
        )
        stats_with_blocks = network_with_blocks.get_deduplication_stats()

        # With blocks, there should be no similarities found (each doc in separate block)
        assert stats_with_blocks["similarity_pairs"] == 0
        # Without blocks, there should be many similarities (all docs are similar)
        assert stats_no_blocks["similarity_pairs"] > 0


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
        embedding_model="all-MiniLM-L6-v2", verbose=False, thresh=0.8
    )

    representatives = network.fit_transform(docs)

    # Basic sanity checks
    assert len(representatives) <= len(docs)
    assert network.is_fitted_ is True
    stats = network.get_deduplication_stats()
    assert stats["original_count"] == len(docs)
    assert isinstance(stats, dict)
