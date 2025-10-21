from typing import Dict, List, Optional, Union, Literal
import logging

import networkx as nx
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

MetricType = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]


class SemanticNetwork:
    """
    A semantic network for document deduplication using embeddings and graph clustering.

    This class follows the scikit-learn pattern with fit(), transform(), and fit_transform() methods.

    The fitting process builds semantic networks from text documents by:
    1. Creating embeddings using sentence transformers
    2. Building an approximate nearest neighbor index for fast similarity search
    3. Constructing a graph where edges represent semantic similarity

    The transformation process identifies duplicate groups and returns representatives.

    Attributes:
        embedding_model_name: Name/path of sentence transformer model
        metric: Distance metric for Annoy index
        n_trees: Number of trees for Annoy index
        thresh: Similarity threshold for connecting documents
        top_k: Maximum neighbors to check per document
        verbose: Whether to show progress bars and detailed logging
        is_fitted_: Whether the model has been fitted
        embeddings_: Document embeddings array (available after fitting)
        index_: Annoy index for similarity search (available after fitting)
        graph_: NetworkX graph of document similarities (available after fitting)
        neighbor_data_: DataFrame of pairwise similarities (available after fitting)
        blocks_: Block assignments for documents (available after fitting with blocks)
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        metric: MetricType = "angular",
        n_trees: int = 10,
        thresh: float = 0.7,
        top_k: int = 100,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the SemanticNetwork.

        Args:
            embedding_model: Name/path of sentence transformer model to use
            metric: Distance metric for Annoy index ('angular', 'euclidean', etc.)
            n_trees: Number of trees for Annoy index (more = better accuracy, slower build)
            thresh: Default similarity threshold for connecting documents (0.0 to 1.0)
            top_k: Default maximum number of neighbors to check per document
            verbose: Whether to show progress bars and detailed logging
        """
        self.embedding_model_name = embedding_model
        self.metric = metric
        self.n_trees = n_trees
        self.thresh = thresh
        self.top_k = top_k
        self.verbose = verbose

        # Fitted state
        self.is_fitted_ = False
        self.embeddings_: Optional[np.ndarray] = None
        self.index_: Optional[AnnoyIndex] = None
        self.graph_: Optional[nx.Graph] = None
        self.neighbor_data_: Optional[pd.DataFrame] = None

        # Training data (stored during fit)
        self._docs: Optional[List[str]] = None
        self._weights: Optional[List[Union[float, int]]] = None
        self.blocks_: Optional[np.ndarray] = None

    def fit(
        self,
        X: List[str],
        y=None,
        weights: Optional[List[Union[float, int]]] = None,
        embeddings: Optional[np.ndarray] = None,
        blocks: Optional[Union[List, np.ndarray]] = None,
    ) -> "SemanticNetwork":
        """
        Learn the semantic relationships between documents.

        This method builds embeddings, creates a similarity index, finds pairwise
        similarities, and constructs the semantic graph.

        Args:
            X: List of text documents to learn from
            y: Ignored, present for API compatibility
            weights: Optional list of weights for document importance.
                    Higher weights = more likely to be chosen as representative.
                    Must be same length as X if provided.
            embeddings: Optional pre-computed embeddings array with shape (len(X), embedding_dim).
                       If provided, document embedding generation will be skipped.
            blocks: Optional blocking variable(s) for documents. Can be:
                   - List/array of strings or ints for single blocking variable
                   - 2D array for multiple blocking variables (shape: len(X), n_block_vars)
                   Only documents within the same block(s) will be compared for similarity.

        Returns:
            self: Returns the fitted estimator

        Raises:
            ValueError: If weights provided but length doesn't match X
            ValueError: If embeddings provided but shape doesn't match X
            ValueError: If blocks provided but length doesn't match X
        """
        if weights is not None and len(weights) != len(X):
            raise ValueError(
                f"Weights length ({len(weights)}) must match X length ({len(X)})"
            )

        if embeddings is not None and embeddings.shape[0] != len(X):
            raise ValueError(
                f"Embeddings shape[0] ({embeddings.shape[0]}) must match X length ({len(X)})"
            )

        # Validate and process blocks
        if blocks is not None:
            blocks_array = np.array(blocks)
            if blocks_array.ndim == 1:
                if len(blocks_array) != len(X):
                    raise ValueError(
                        f"Blocks length ({len(blocks_array)}) must match X length ({len(X)})"
                    )
                # Reshape to 2D for consistent handling
                blocks_array = blocks_array.reshape(-1, 1)
            elif blocks_array.ndim == 2:
                if blocks_array.shape[0] != len(X):
                    raise ValueError(
                        f"Blocks shape[0] ({blocks_array.shape[0]}) must match X length ({len(X)})"
                    )
            else:
                raise ValueError("Blocks must be 1D or 2D array")

            self.blocks_ = blocks_array
            if self.verbose:
                n_vars = blocks_array.shape[1]
                n_unique_blocks = len(np.unique(blocks_array.view(np.void), axis=0))
                logger.info(
                    f"Using {n_vars} blocking variable(s) with {n_unique_blocks} unique block(s)"
                )

        # Store training data
        self._docs = X
        self._weights = weights  # Store custom embeddings if provided
        if embeddings is not None:
            self.embeddings_ = embeddings
            if self.verbose:
                logger.info(
                    f"Using provided embeddings with shape: {self.embeddings_.shape}"
                )

        if self.verbose:
            logger.info(f"Fitting SemanticNetwork on {len(X)} documents")

        # Build the semantic network
        if embeddings is None:
            self._embed_documents()
        self._build_vector_index()
        self._get_pairwise_similarities()
        self._build_graph()

        self.is_fitted_ = True

        if self.verbose:
            logger.info("Fitting complete")

        return self

    def transform(
        self, X: Optional[List[str]] = None, return_representatives: bool = True
    ) -> Union[List[str], Dict[int, int]]:
        """
        Apply deduplication to documents.

        Args:
            X: Optional list of documents to transform. If None, uses the documents from fit().
            return_representatives: If True, return list of representative documents.
                                  If False, return mapping dict from document index to representative index.

        Returns:
            Either a list of representative documents or a mapping dictionary

        Raises:
            ValueError: If the model hasn't been fitted yet
            ValueError: If X is provided and doesn't match the fitted documents
        """
        if not self.is_fitted_:
            raise ValueError(
                "This SemanticNetwork instance is not fitted yet. Call 'fit' first."
            )

        if self._docs is None:
            raise ValueError(
                "No training documents found. This should not happen after fitting."
            )

        if X is not None:
            if X != self._docs:
                raise ValueError(
                    "Transform X must match the documents used in fit(). "
                    "Use fit_transform() if you want to fit and transform different documents."
                )

        # Get deduplication mapping
        mapping = self._get_deduplication_mapping()

        if return_representatives:
            # Return representative documents
            mapped_indices = set(mapping.keys())
            representative_indices = [
                i for i in range(len(self._docs)) if i not in mapped_indices
            ]
            return [self._docs[i] for i in representative_indices]
        else:
            # Return the mapping
            return mapping

    def fit_transform(
        self,
        X: List[str],
        y=None,
        weights: Optional[List[Union[float, int]]] = None,
        embeddings: Optional[np.ndarray] = None,
        blocks: Optional[Union[List, np.ndarray]] = None,
        return_representatives: bool = True,
    ) -> Union[List[str], Dict[int, int]]:
        """
        Fit the model and transform the documents in one step.

        Args:
            X: List of text documents to process
            y: Ignored, present for API compatibility
            weights: Optional list of weights for document importance
            embeddings: Optional pre-computed embeddings array with shape (len(X), embedding_dim).
                       If provided, document embedding generation will be skipped.
            blocks: Optional blocking variable(s) for documents. Only documents within
                   the same block(s) will be compared for similarity.
            return_representatives: If True, return list of representative documents.
                                  If False, return mapping dict.

        Returns:
            Either a list of representative documents or a mapping dictionary
        """
        return self.fit(X, y, weights, embeddings, blocks).transform(
            return_representatives=return_representatives
        )

    def get_deduplication_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about the deduplication results.

        Returns:
            Dictionary containing deduplication statistics

        Raises:
            ValueError: If the model hasn't been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError(
                "This SemanticNetwork instance is not fitted yet. Call 'fit' first."
            )

        if self._docs is None:
            raise ValueError(
                "No training documents found. This should not happen after fitting."
            )

        mapping = self._get_deduplication_mapping()

        original_count = len(self._docs)
        deduplicated_count = len(self.transform())
        reduction_ratio = (
            (original_count - deduplicated_count) / original_count
            if original_count > 0
            else 0
        )

        return {
            "original_count": original_count,
            "deduplicated_count": deduplicated_count,
            "duplicates_found": len(mapping),
            "reduction_ratio": reduction_ratio,
            "similarity_pairs": (
                len(self.neighbor_data_) if self.neighbor_data_ is not None else 0
            ),
            "connected_components": (
                nx.number_connected_components(self.graph_) if self.graph_ else 0
            ),
        }

    def get_duplicate_groups(self) -> List[List[str]]:
        """
        Get groups of duplicate documents as lists of document text.

        Returns:
            List of groups, where each group contains the text of similar documents.
            Groups with only one document (no duplicates) are excluded.

        Raises:
            ValueError: If the model hasn't been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError(
                "This SemanticNetwork instance is not fitted yet. Call 'fit' first."
            )

        if self.graph_ is None:
            raise ValueError("Graph not found. This should not happen after fitting.")

        if self._docs is None:
            raise ValueError(
                "No training documents found. This should not happen after fitting."
            )

        components = list(nx.connected_components(self.graph_))
        groups = []

        for component in components:
            if len(component) > 1:  # Only include groups with duplicates
                group_docs = [self._docs[i] for i in sorted(component)]
                groups.append(group_docs)

        # Sort by group size (largest first)
        groups.sort(key=len, reverse=True)

        if self.verbose:
            logger.info(f"Found {len(groups)} duplicate groups")

        return groups

    def _embed_documents(self) -> np.ndarray:
        """
        Generate embeddings for all documents using the sentence transformer model.

        Returns:
            Document embeddings array of shape (n_docs, embedding_dim)

        Note:
            The embeddings are stored in self.embeddings_ and also returned.
        """
        if self._docs is None:
            raise ValueError("No training documents found. Call fit() first.")

        if self.verbose:
            logger.info(f"Generating embeddings for {len(self._docs)} documents")

        # If embeddings already exist, return them
        if self.embeddings_ is not None:
            if self.verbose:
                logger.info(
                    f"Using existing embeddings with shape: {self.embeddings_.shape}"
                )
            return self.embeddings_

        if self.verbose:
            logger.info(f"Generating embeddings for {len(self._docs)} documents")

        # Use model's built-in progress bar if verbose, otherwise disable
        embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embeddings_ = embedding_model.encode(
            self._docs, show_progress_bar=self.verbose
        )
        # Free up memory by deleting the model reference
        del embedding_model

        if self.verbose:
            logger.info(f"Generated embeddings with shape: {self.embeddings_.shape}")

        return self.embeddings_

    def _build_vector_index(self) -> AnnoyIndex:
        """
        Build an Annoy index for fast approximate nearest neighbor search.

        Returns:
            The built Annoy index

        Raises:
            ValueError: If embeddings haven't been generated yet

        Note:
            The index is stored in self.index_ and also returned.
        """
        if self.embeddings_ is None:
            raise ValueError(
                "Embeddings not found. Please run _embed_documents() first."
            )

        embedding_dim = self.embeddings_.shape[1]
        self.index_ = AnnoyIndex(embedding_dim, self.metric)  # type: ignore

        if self.verbose:
            logger.info(
                f"Building Annoy index with {self.n_trees} trees for {len(self.embeddings_)} embeddings"
            )
            iterator = tqdm(
                enumerate(self.embeddings_),
                total=len(self.embeddings_),
                desc="Adding embeddings to index",
            )
        else:
            iterator = enumerate(self.embeddings_)

        for i, embedding_vector in iterator:
            self.index_.add_item(i, embedding_vector)

        if self.verbose:
            logger.info("Building index trees...")
        self.index_.build(self.n_trees)

        if self.verbose:
            logger.info("Vector index built successfully")

        return self.index_

    def _get_pairwise_similarities(self) -> pd.DataFrame:
        """
        Find pairwise similarities between documents above a threshold.

        Uses the Annoy index to efficiently find nearest neighbors for each document,
        then calculates exact similarities and filters by threshold. If blocks are
        provided, only compares documents within the same block(s).

        Returns:
            DataFrame of similarities

        Raises:
            ValueError: If embeddings or index haven't been built yet

        Note:
            Similarity is calculated as (1 - angular_distance) for angular metric.
            Results include columns: source_idx, target_idx, similarity, source_name, target_name
        """
        if self.embeddings_ is None or self.index_ is None:
            raise ValueError(
                "Embeddings or index not found. Please run _embed_documents() and _build_vector_index() first."
            )

        if self._docs is None:
            raise ValueError("No training documents found. Call fit() first.")

        if self.verbose:
            logger.info(
                f"Finding pairwise similarities with threshold {self.thresh}, checking top {self.top_k} neighbors"
            )

        results = []

        if self.blocks_ is not None:
            # Use blocking: only compare documents within the same block(s)
            if self.verbose:
                logger.info("Using blocking for similarity search")

            # Group documents by their block values
            block_groups = {}
            for idx in range(len(self._docs)):
                # Convert block values to tuple for hashing
                block_key = tuple(self.blocks_[idx])
                if block_key not in block_groups:
                    block_groups[block_key] = []
                block_groups[block_key].append(idx)

            if self.verbose:
                logger.info(f"Created {len(block_groups)} blocks for comparison")
                iterator = tqdm(block_groups.items(), desc="Processing blocks")
            else:
                iterator = block_groups.items()

            for block_key, block_indices in iterator:
                # Only compare documents within this block
                for i, idx_source in enumerate(block_indices):
                    # Calculate similarities with all other documents in the same block
                    for idx_target in block_indices[i + 1 :]:  # Avoid duplicate pairs
                        # Calculate cosine similarity directly from embeddings
                        embedding_source = self.embeddings_[idx_source]
                        embedding_target = self.embeddings_[idx_target]

                        # Cosine similarity for normalized vectors
                        similarity = np.dot(embedding_source, embedding_target)

                        if similarity >= self.thresh:
                            # Add both directions for consistency with non-blocking approach
                            for src, tgt in [
                                (idx_source, idx_target),
                                (idx_target, idx_source),
                            ]:
                                result_dict = {
                                    "source_idx": src,
                                    "target_idx": tgt,
                                    "similarity": similarity,
                                    "source_name": self._docs[src],
                                    "target_name": self._docs[tgt],
                                }
                                results.append(result_dict)
        else:
            # Original approach: use Annoy index for all documents
            if self.verbose:
                iterator = tqdm(
                    range(len(self.embeddings_)), desc="Finding similarities"
                )
            else:
                iterator = range(len(self.embeddings_))

            for idx_source in iterator:
                neighbors = self.index_.get_nns_by_item(
                    idx_source, self.top_k, include_distances=True
                )

                for idx_target, dist in zip(*neighbors):
                    similarity = 1 - dist  # Convert angular distance to similarity

                    if idx_source != idx_target and similarity >= self.thresh:
                        result_dict = {
                            "source_idx": idx_source,
                            "target_idx": idx_target,
                            "similarity": similarity,
                            "source_name": self._docs[idx_source],
                            "target_name": self._docs[idx_target],
                        }
                        results.append(result_dict)

        self.neighbor_data_ = pd.DataFrame(results)

        if self.verbose:
            logger.info(
                f"Found {len(self.neighbor_data_)} similarity pairs above threshold {self.thresh}"
            )

        return self.neighbor_data_

    def _build_graph(self) -> nx.Graph:
        """
        Build a NetworkX graph from pairwise similarities.

        Creates a graph where:
        - Nodes represent documents (with 'name' and 'weight' attributes)
        - Edges represent similarities above the threshold (with 'similarity' attribute)

        Returns:
            The constructed NetworkX graph

        Raises:
            ValueError: If neighbor data hasn't been computed yet

        Note:
            Node weights default to 1.0 if no weights were provided during initialization.
            The graph is stored in self.graph_ and also returned.
        """
        if self.neighbor_data_ is None:
            raise ValueError(
                "Neighbor data not found. Please run _get_pairwise_similarities() first."
            )

        if self._docs is None:
            raise ValueError("No training documents found. Call fit() first.")

        if self.verbose:
            logger.info(
                f"Building graph from {len(self.neighbor_data_)} similarity edges"
            )

        # Create graph from edge list
        if len(self.neighbor_data_) > 0:
            graph = nx.from_pandas_edgelist(
                self.neighbor_data_,
                source="source_idx",
                target="target_idx",
                edge_attr="similarity",
            )
        else:
            # No similarities found, create empty graph
            graph = nx.Graph()

        # Create weight mapping
        weight_dict = {}
        if self._weights is not None:
            weight_dict = dict(zip(range(len(self._docs)), self._weights))

        # Update nodes with names and weights
        for node in graph.nodes:
            graph.nodes[node]["name"] = self._docs[node]
            graph.nodes[node]["weight"] = weight_dict.get(node, 1.0)  # Default to 1.0

        # Add isolated nodes (documents with no similarities above threshold)
        for i in range(len(self._docs)):
            if i not in graph.nodes:
                graph.add_node(i, name=self._docs[i], weight=weight_dict.get(i, 1.0))

        self.graph_ = graph

        if self.verbose:
            num_components = nx.number_connected_components(graph)
            logger.info(
                f"Built graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, {num_components} components"
            )

        return graph

    def _get_deduplication_mapping(self) -> Dict[int, int]:
        """
        Get a mapping from document indices to their representative indices.

        For each connected component (group of similar documents), selects the
        document with the highest weight as the representative. All other documents
        in the component are mapped to this representative.

        Returns:
            Dictionary mapping document index -> representative index.
            Only contains entries for documents that should be deduplicated
            (i.e., representatives are not included in the mapping).

        Raises:
            ValueError: If graph hasn't been built yet

        Example:
            If docs [0, 1, 2] are similar and doc 1 has highest weight,
            returns {0: 1, 2: 1} (doc 1 is not in the mapping as it's the representative)
        """
        if self.graph_ is None:
            raise ValueError("Graph not found. Please run _build_graph() first.")

        if self._docs is None:
            raise ValueError("No training documents found. Call fit() first.")

        components = list(nx.connected_components(self.graph_))

        if self.verbose:
            logger.info(
                f"Found {len(components)} connected components for deduplication"
            )

        mapping = {}
        for component in components:
            if len(component) > 1:  # Only process components with multiple nodes
                subgraph = self.graph_.subgraph(component)
                heaviest_node = max(
                    subgraph.nodes, key=lambda n: subgraph.nodes[n]["weight"]
                )

                # Map all other nodes in component to the heaviest node
                for node in subgraph.nodes:
                    if node != heaviest_node:
                        mapping[node] = heaviest_node

                if self.verbose and len(component) > 1:
                    component_docs = [self._docs[i] for i in component]
                    representative_doc = self._docs[heaviest_node]
                    logger.info(
                        f"Component of {len(component)} docs represented by: '{representative_doc[:50]}...'"
                    )

        if self.verbose:
            logger.info(
                f"Created mapping for {len(mapping)} documents to be deduplicated"
            )

        return mapping
