from typing import Dict, List, Optional, Union, Literal
import logging

import networkx as nx
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

MetricType = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]


class SemanticNetwork:
    """
    A semantic network for document deduplication using embeddings and graph clustering.

    This class follows the scikit-learn pattern with fit(), transform(), and fit_transform() methods.
    Users must provide pre-computed embeddings during the fit process.

    The fitting process builds semantic networks from text documents by:
    1. Using provided embeddings
    2. Building an approximate nearest neighbor index for fast similarity search
    3. Constructing a graph where edges represent semantic similarity

    The transformation process identifies duplicate groups and returns representatives.

    Attributes:
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
        metric: MetricType = "angular",
        n_trees: int = 10,
        thresh: float = 0.7,
        top_k: int = 100,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the SemanticNetwork.

        Args:
            metric: Distance metric for Annoy index ('angular', 'euclidean', etc.)
            n_trees: Number of trees for Annoy index (more = better accuracy, slower build)
            thresh: Default similarity threshold for connecting documents (0.0 to 1.0)
            top_k: Default maximum number of neighbors to check per document
            verbose: Whether to show progress bars and detailed logging
        """
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
        self._labels: Optional[List[str]] = None
        self._ids: Optional[List[Union[str, int]]] = None
        self._node_data: Optional[Dict] = None
        self._weights: Optional[List[Union[float, int]]] = None
        self.blocks_: Optional[np.ndarray] = None

    def fit(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        ids: Optional[List[Union[str, int]]] = None,
        node_data: Optional[Dict] = None,
        weights: Optional[List[Union[float, int]]] = None,
        blocks: Optional[Union[List, np.ndarray]] = None,
    ) -> "SemanticNetwork":
        """
        Learn the semantic relationships between documents.

        This method uses provided embeddings to create a similarity index, finds pairwise
        similarities, and constructs the semantic graph.

        Args:
            embeddings: Pre-computed embeddings array with shape (n_docs, embedding_dim).
                       Must be provided - this class does not generate embeddings.
            labels: Optional list of text labels/documents for the embeddings.
                   If not provided, will use string indices as labels.
            ids: Optional list of custom IDs for the embeddings.
                If not provided, will use integer indices as IDs.
            node_data: Optional dictionary containing additional data to attach to nodes.
                      Keys should be node attributes, values should be lists of same length as embeddings.
            weights: Optional list of weights for document importance.
                    Higher weights = more likely to be chosen as representative.
                    Must be same length as embeddings if provided.
            blocks: Optional blocking variable(s) for documents. Can be:
                   - List/array of strings or ints for single blocking variable
                   - 2D array for multiple blocking variables (shape: n_docs, n_block_vars)
                   Only documents within the same block(s) will be compared for similarity.

        Returns:
            self: Returns the fitted estimator

        Raises:
            ValueError: If weights provided but length doesn't match embeddings
            ValueError: If labels provided but length doesn't match embeddings
            ValueError: If ids provided but length doesn't match embeddings
            ValueError: If blocks provided but length doesn't match embeddings
            ValueError: If node_data values don't match embeddings length
        """
        n_docs = embeddings.shape[0]

        if weights is not None and len(weights) != n_docs:
            raise ValueError(
                f"Weights length ({len(weights)}) must match embeddings length ({n_docs})"
            )

        if labels is not None and len(labels) != n_docs:
            raise ValueError(
                f"Labels length ({len(labels)}) must match embeddings length ({n_docs})"
            )

        if ids is not None and len(ids) != n_docs:
            raise ValueError(
                f"IDs length ({len(ids)}) must match embeddings length ({n_docs})"
            )

        if node_data is not None:
            for key, values in node_data.items():
                if len(values) != n_docs:
                    raise ValueError(
                        f"Node data '{key}' length ({len(values)}) must match embeddings length ({n_docs})"
                    )

        # Validate and process blocks
        if blocks is not None:
            blocks_array = np.array(blocks)
            if blocks_array.ndim == 1:
                if len(blocks_array) != n_docs:
                    raise ValueError(
                        f"Blocks length ({len(blocks_array)}) must match embeddings length ({n_docs})"
                    )
                # Reshape to 2D for consistent handling
                blocks_array = blocks_array.reshape(-1, 1)
            elif blocks_array.ndim == 2:
                if blocks_array.shape[0] != n_docs:
                    raise ValueError(
                        f"Blocks shape[0] ({blocks_array.shape[0]}) must match embeddings length ({n_docs})"
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
        self._labels = labels if labels is not None else [str(i) for i in range(n_docs)]
        self._ids = ids if ids is not None else list(range(n_docs))
        self._node_data = node_data
        self._weights = weights
        self.embeddings_ = embeddings
        
        if self.verbose:
            logger.info(
                f"Using provided embeddings with shape: {self.embeddings_.shape}"
            )
            logger.info(f"Fitting SemanticNetwork on {n_docs} documents")        # Build the semantic network
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

        if self._labels is None:
            raise ValueError(
                "No training documents found. This should not happen after fitting."
            )

        if X is not None:
            if X != self._labels:
                raise ValueError(
                    "Transform X must match the labels used in fit(). "
                    "Use fit_transform() if you want to fit and transform different documents."
                )

        # Get deduplication mapping
        mapping = self._get_deduplication_mapping()

        if return_representatives:
            # Return representative documents
            mapped_indices = set(mapping.keys())
            representative_indices = [
                i for i in range(len(self._labels)) if i not in mapped_indices
            ]
            return [self._labels[i] for i in representative_indices]
        else:
            # Return the mapping
            return mapping

    def fit_transform(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        ids: Optional[List[Union[str, int]]] = None,
        node_data: Optional[Dict] = None,
        weights: Optional[List[Union[float, int]]] = None,
        blocks: Optional[Union[List, np.ndarray]] = None,
        return_representatives: bool = True,
    ) -> Union[List[str], Dict[int, int]]:
        """
        Fit the model and transform the documents in one step.

        Args:
            embeddings: Pre-computed embeddings array with shape (n_docs, embedding_dim).
                       Must be provided - this class does not generate embeddings.
            labels: Optional list of text labels/documents for the embeddings.
            ids: Optional list of custom IDs for the embeddings.
            node_data: Optional dictionary containing additional data to attach to nodes.
            weights: Optional list of weights for document importance
            blocks: Optional blocking variable(s) for documents. Only documents within
                   the same block(s) will be compared for similarity.
            return_representatives: If True, return list of representative documents.
                                  If False, return mapping dict.

        Returns:
            Either a list of representative documents or a mapping dictionary
        """
        return self.fit(embeddings, labels, ids, node_data, weights, blocks).transform(
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

        if self._labels is None:
            raise ValueError(
                "No training documents found. This should not happen after fitting."
            )

        mapping = self._get_deduplication_mapping()

        original_count = len(self._labels)
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

        if self._labels is None:
            raise ValueError(
                "No training documents found. This should not happen after fitting."
            )

        components = list(nx.connected_components(self.graph_))
        groups = []

        for component in components:
            if len(component) > 1:  # Only include groups with duplicates
                group_docs = [self._labels[i] for i in sorted(component)]
                groups.append(group_docs)

        # Sort by group size (largest first)
        groups.sort(key=len, reverse=True)

        if self.verbose:
            logger.info(f"Found {len(groups)} duplicate groups")

        return groups

    def _build_vector_index(self) -> AnnoyIndex:
        """
        Build an Annoy index for fast approximate nearest neighbor search.

        Returns:
            The built Annoy index

        Raises:
            ValueError: If embeddings haven't been provided yet

        Note:
            The index is stored in self.index_ and also returned.
        """
        if self.embeddings_ is None:
            raise ValueError(
                "Embeddings not found. Please provide embeddings in fit() method."
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
                "Embeddings or index not found. Please provide embeddings in fit() method and run _build_vector_index() first."
            )

        if self._labels is None:
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
            for idx in range(len(self._labels)):
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
                                    "source_name": self._labels[src],
                                    "target_name": self._labels[tgt],
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
                            "source_name": self._labels[idx_source],
                            "target_name": self._labels[idx_target],
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

        if self._labels is None:
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
            weight_dict = dict(zip(range(len(self._labels)), self._weights))

        # Update nodes with names, weights, and additional data
        for node in graph.nodes:
            # Set name from labels
            graph.nodes[node]["name"] = self._labels[node]
            # Set weight
            graph.nodes[node]["weight"] = weight_dict.get(node, 1.0)  # Default to 1.0
            # Set custom ID if provided
            if self._ids is not None:
                graph.nodes[node]["id"] = self._ids[node]
            # Set additional node data if provided
            if self._node_data is not None:
                for key, values in self._node_data.items():
                    graph.nodes[node][key] = values[node]

        # Add isolated nodes (documents with no similarities above threshold)
        for i in range(len(self._labels)):
            if i not in graph.nodes:
                graph.add_node(i, name=self._labels[i], weight=weight_dict.get(i, 1.0))
                if self._ids is not None:
                    graph.nodes[i]["id"] = self._ids[i]
                if self._node_data is not None:
                    for key, values in self._node_data.items():
                        graph.nodes[i][key] = values[i]

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

        if self._labels is None:
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
                    representative_doc = self._labels[heaviest_node]
                    logger.info(
                        f"Component of {len(component)} docs represented by: '{representative_doc[:50]}...'"
                    )

        if self.verbose:
            logger.info(
                f"Created mapping for {len(mapping)} documents to be deduplicated"
            )

        return mapping
