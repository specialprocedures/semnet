import logging
from typing import Dict, List, Literal, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

MetricType = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]


class SemanticNetwork:
    """
    A semantic network builder for creating graphs from document embeddings.

    This class follows the scikit-learn pattern with fit() and transform() methods.
    Users must provide pre-computed embeddings during the fit process.

    The fitting process builds an approximate nearest neighbor index from embeddings.
    The transformation process constructs a graph where edges represent semantic similarity.

    Key Methods:
        fit(): Build the similarity index from provided embeddings
        transform(): Construct and return a networkx object
        fit_transform(): Combined fit and transform in one step
        to_pandas(): Export graph structure to pandas DataFrames for analysis

    Attributes:
        metric: Distance metric for the Annoy index
        n_trees: Number of trees for the Annoy index
        thresh: Similarity threshold for connecting documents
        top_k: Maximum neighbors to check per document
        verbose: Whether to show progress bars and detailed logging
        is_fitted_: Whether the model has been fitted
        embeddings_: Document embeddings array (available after fitting)
        index_: Annoy index for similarity search (available after fitting)
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
            thresh: Similarity threshold for connecting documents (0.0 to 1.0)
            top_k: Maximum number of neighbors to check per document
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

        # Training data (stored during fit)
        self._labels: Optional[List[str]] = None
        self._node_data: Optional[Dict] = None

    def fit(
        self,
        embeddings: np.ndarray,
    ) -> "SemanticNetwork":
        """
        Build the index from document embeddings.

        This method uses provided embeddings to create an Annoy index for
        fast nearest neighbor search.

        Args:
            embeddings: Pre-computed embeddings array with shape (n_docs, embedding_dim).
            labels: Optional list of text labels/documents for the embeddings.
                   If not provided, will use string indices as labels.
            node_data: Optional dictionary containing additional data to attach to nodes.
                      Format: {node_index: {attribute_name: value, ...}, ...}
                      OR {node_index: single_value, ...} (will be stored as {'value': single_value})
                      Only nodes present in the dictionary will get additional attributes.

        Returns:
            self: Returns the fitted estimator

        Raises:
            ValueError: If labels provided but length doesn't match embeddings
            ValueError: If ids provided but length doesn't match embeddings
            ValueError: If node_data values don't match embeddings length
        """

        self.embeddings_ = embeddings

        if self.verbose:
            logger.info(
                f"Using provided embeddings with shape: {self.embeddings_.shape}"
            )
            logger.info(f"Fitting SemanticNetwork on {len(embeddings)} documents")

        # Build the vector index
        self._build_vector_index()

        self.is_fitted_ = True

        if self.verbose:
            logger.info("Fitting complete")

        return self

    def transform(
        self,
        thresh: Optional[float] = None,
        top_k: Optional[int] = None,
        labels: Optional[List[str]] = None,
        node_data: Optional[Dict] = None,
    ) -> nx.Graph:
        """
        Build and return a weighted graph from the fitted embeddings.

        Args:
            thresh: The similarity threshold for edge inclusion.
                   If None, uses the threshold from initialization.
            top_k: Optional max neighbors override for this transform.
                  If None, uses the top_k from initialization.

        Returns:
            NetworkX graph where nodes represent documents and edges represent
            similarities above the threshold.

        Raises:
            ValueError: If the model hasn't been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError(
                "This SemanticNetwork instance is not fitted yet. Call 'fit' first."
            )

        n_docs = self.embeddings_.shape[0]

        if labels is not None and len(labels) != n_docs:
            raise ValueError(
                f"Labels length ({len(labels)}) must match embeddings length ({n_docs})"
            )

        if node_data is not None:
            # Validate node_data format: should be {node_index: {attribute_dict}} or {node_index: value}
            if not isinstance(node_data, dict):
                raise ValueError("Node data must be a dictionary")

            # Check if all keys are integers (node indices)
            non_integer_keys = [
                k for k in node_data.keys() if not isinstance(k, (int, np.integer))
            ]
            if non_integer_keys:
                raise ValueError(
                    f"Node data keys must be integer node indices, got: {non_integer_keys}"
                )

            # Validate that node_data keys are valid node indices
            invalid_indices = [
                idx for idx in node_data.keys() if idx >= n_docs or idx < 0
            ]
            if invalid_indices:
                raise ValueError(
                    f"Node data contains invalid indices {invalid_indices}. Indices must be 0 <= idx < {n_docs}"
                )

            # Convert single values to dictionary format for consistency
            # If values are not dictionaries, wrap them in a dictionary with 'value' key
            processed_node_data = {}
            for k, v in node_data.items():
                if isinstance(v, dict):
                    processed_node_data[k] = v
                else:
                    processed_node_data[k] = {"value": v}
            node_data = processed_node_data

        # Store training data
        self._labels = labels if labels is not None else [str(i) for i in range(n_docs)]
        self._node_data = node_data

        # Use provided thresholds or fall back to instance defaults
        effective_thresh = thresh if thresh is not None else self.thresh
        effective_top_k = top_k if top_k is not None else self.top_k

        # Get pairwise similarities
        neighbor_data = self._get_pairwise_similarities(
            effective_thresh, effective_top_k
        )

        # Build and return the graph
        return self._build_graph(neighbor_data)

    def fit_transform(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        node_data: Optional[Dict] = None,
        thresh: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> nx.Graph:
        """
        Fit the model and transform the embeddings in one step.

        Args:
            embeddings: Pre-computed embeddings array with shape (n_docs, embedding_dim).
            labels: Optional list of text labels/documents for the embeddings.
            node_data: Optional dictionary containing additional data to attach to nodes.
            thresh: Optional similarity threshold override for this transform.
            top_k: Optional max neighbors override for this transform.

        Returns:
            NetworkX graph representing the semantic network
        """
        return self.fit(embeddings=embeddings).transform(
            thresh=thresh, top_k=top_k, labels=labels, node_data=node_data
        )

    def to_pandas(
        self, graph: Optional[nx.Graph] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export a NetworkX graph to pandas DataFrames.

        By default, exports the most recently transformed graph. Optionally accepts
        an arbitrary NetworkX graph (useful for subgraphs or modified graphs).

        Args:
            graph: Optional NetworkX graph to export. If None, will raise an error
                  since no graph is stored by default after transform.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - nodes (pd.DataFrame): Node attributes with index as node ID.
                  Columns include all node attributes from the graph.
                - edges (pd.DataFrame): Edge list with columns 'source', 'target',
                  and any edge attributes (e.g., 'weight').

        Raises:
            ValueError: If no graph is provided and the model hasn't been fitted yet

        Examples:
            >>> # Build and export a graph
            >>> network = SemanticNetwork(thresh=0.8)
            >>> graph = network.fit_transform(embeddings, labels=docs)
            >>> nodes, edges = network.to_pandas(graph)

            >>> # Export a subgraph
            >>> subgraph = graph.subgraph([0, 1, 2])
            >>> sub_nodes, sub_edges = network.to_pandas(subgraph)
        """
        if graph is None:
            raise ValueError(
                "No graph provided. Call transform() to get a graph, then pass it to to_pandas()."
            )

        # Convert nodes to DataFrame
        nodes = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")

        # Convert edges to DataFrame
        if graph.number_of_edges() > 0:
            edges = nx.to_pandas_edgelist(graph)
        else:
            # Create empty DataFrame with expected columns if no edges
            edges = pd.DataFrame(columns=["source", "target"])

        return nodes, edges

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

    def _get_pairwise_similarities(self, thresh: float, top_k: int) -> pd.DataFrame:
        """
        Find pairwise similarities between documents above a threshold.

        Uses the Annoy index to efficiently find nearest neighbors for each document,
        then calculates exact similarities and filters by threshold.

        Args:
            thresh: Similarity threshold for including edges
            top_k: Maximum number of neighbors to check per document

        Returns:
            DataFrame of similarities with columns: source_idx, target_idx, weight, source_name, target_name

        Raises:
            ValueError: If embeddings or index haven't been built yet
        """
        if self.embeddings_ is None or self.index_ is None:
            raise ValueError(
                "Embeddings or index not found. Please provide embeddings in fit() method and run _build_vector_index() first."
            )

        if self._labels is None:
            raise ValueError("No training documents found. Call fit() first.")

        if self.verbose:
            logger.info(
                f"Finding pairwise similarities with threshold {thresh}, checking top {top_k} neighbors"
            )

        results = []

        if self.verbose:
            iterator = tqdm(range(len(self.embeddings_)), desc="Finding similarities")
        else:
            iterator = range(len(self.embeddings_))

        for idx_source in iterator:
            neighbors = self.index_.get_nns_by_item(
                idx_source, top_k, include_distances=True
            )

            for idx_target, dist in zip(*neighbors):
                similarity = 1 - dist  # Convert angular distance to similarity

                if idx_source != idx_target and similarity >= thresh:
                    result_dict = {
                        "source_idx": idx_source,
                        "target_idx": idx_target,
                        "weight": similarity,
                        "source_name": self._labels[idx_source],
                        "target_name": self._labels[idx_target],
                    }
                    results.append(result_dict)

        neighbor_data = pd.DataFrame(results)

        if self.verbose:
            logger.info(
                f"Found {len(neighbor_data)} similarity pairs above threshold {thresh}"
            )

        return neighbor_data

    def _build_graph(self, neighbor_data: pd.DataFrame) -> nx.Graph:
        """
        Build a NetworkX graph from pairwise similarities.

        Creates a graph where:
        - Nodes represent documents
        - Edges represent similarities above the threshold (with 'weight' attribute representing similarity)

        Args:
            neighbor_data: DataFrame of pairwise similarities

        Returns:
            The constructed NetworkX graph

        Raises:
            ValueError: If training data hasn't been set

        Note:
            The graph includes all documents as nodes, even if they have no similarities above threshold.
        """

        if self._labels is None:
            raise ValueError("No training documents found. Call fit() first.")

        if self.verbose:
            logger.info(f"Building graph from {len(neighbor_data)} similarity edges")

        # Instantiate undirected graph
        G = nx.Graph()

        # Add all nodes with their attributes
        for i in range(len(self._labels)):
            # Set basic attributes
            attrs = {
                "label": self._labels[i],
                "id": i,
            }

            # Add custom node data if provided for this specific node
            if self._node_data is not None and i in self._node_data:
                attrs.update(self._node_data[i])

            G.add_node(i, **attrs)

        # Add edges from neighbor data
        for _, row in neighbor_data.iterrows():
            G.add_edge(
                row["source_idx"],
                row["target_idx"],
                weight=row["weight"],
            )

        if self.verbose:
            num_components = nx.number_connected_components(G)
            logger.info(
                f"Built graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {num_components} components"
            )

        return G
