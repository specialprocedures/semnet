from typing import Dict, List, Optional, Union
import logging

import networkx as nx
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class SemanticNetwork:
    """
    A semantic network for document deduplication using embeddings and graph clustering.

    This class builds semantic networks from text documents by:
    1. Creating embeddings using sentence transformers OR using pre-computed embeddings
    2. Building an approximate nearest neighbor index for fast similarity search
    3. Constructing a graph where edges represent semantic similarity
    4. Finding connected components to identify duplicate groups
    5. Selecting representatives based on weights (importance/frequency)

    Attributes:
        docs: List of input documents
        weights: Optional weights for document importance
        embedding_model: Loaded SentenceTransformer model (None if using pre-computed embeddings)
        embeddings: Document embeddings array
        index: Annoy index for similarity search
        graph: NetworkX graph of document similarities
        neighbor_data: DataFrame of pairwise similarities
        verbose: Whether to show progress bars and detailed logging
    """

    def __init__(
        self,
        docs: List[str],
        weights: Optional[List[Union[float, int]]] = None,
        embeddings: Optional[np.ndarray] = None,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        metric: str = "angular",
        n_trees: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the SemanticNetwork.

        Args:
            docs: List of text documents to process
            weights: Optional list of weights for document importance.
                    Higher weights = more likely to be chosen as representative.
                    Must be same length as docs if provided.
            embeddings: Optional pre-computed embeddings array with shape (len(docs), embedding_dim).
                       If provided, embedding_model will not be loaded and embed_documents() will be skipped.
            embedding_model: Name/path of sentence transformer model to use (ignored if embeddings provided)
            metric: Distance metric for Annoy index ('angular', 'euclidean', etc.)
            n_trees: Number of trees for Annoy index (more = better accuracy, slower build)
            verbose: Whether to show progress bars and detailed logging

        Raises:
            ValueError: If weights provided but length doesn't match docs
            ValueError: If embeddings provided but shape doesn't match docs
        """
        if weights is not None and len(weights) != len(docs):
            raise ValueError(
                f"Weights length ({len(weights)}) must match docs length ({len(docs)})"
            )

        self.docs = docs
        self.weights = weights
        self.n_trees = n_trees
        self.metric = metric
        self.verbose = verbose

        # Initialize model with progress bar if verbose
        if self.verbose:
            logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = embedding_model

        # Initialize state
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[AnnoyIndex] = None
        self.graph: Optional[nx.Graph] = None
        self.neighbor_data: Optional[pd.DataFrame] = None

    def embed_documents(self) -> np.ndarray:
        """
        Generate embeddings for all documents using the sentence transformer model.

        Returns:
            Document embeddings array of shape (n_docs, embedding_dim)

        Note:
            The embeddings are stored in self.embeddings and also returned.
        """
        if self.verbose:
            logger.info(f"Generating embeddings for {len(self.docs)} documents")

        # If embeddings already exist (pre-computed or previously generated), return them
        if self.embeddings is not None:
            if self.verbose:
                logger.info(
                    f"Using existing embeddings with shape: {self.embeddings.shape}"
                )
            return self.embeddings

        # Check if we have a model to generate embeddings
        if self.embedding_model is None:
            raise RuntimeError(
                "No embedding model available and no pre-computed embeddings provided. "
                "Either provide embeddings during initialization or ensure embedding_model is set."
            )

        if self.verbose:
            logger.info(f"Generating embeddings for {len(self.docs)} documents")

        # Use model's built-in progress bar if verbose, otherwise disable
        embedding_model = SentenceTransformer(self.embedding_model)
        self.embeddings = embedding_model.encode(
            self.docs, show_progress_bar=self.verbose
        )
        # Free up memory by deleting the model reference
        del embedding_model

        if self.verbose:
            logger.info(f"Generated embeddings with shape: {self.embeddings.shape}")

        return self.embeddings

    def build_vector_index(self) -> AnnoyIndex:
        """
        Build an Annoy index for fast approximate nearest neighbor search.

        Returns:
            The built Annoy index

        Raises:
            ValueError: If embeddings haven't been generated yet

        Note:
            The index is stored in self.index and also returned.
        """
        if self.embeddings is None:
            raise ValueError(
                "Embeddings not found. Please run embed_documents() first."
            )

        embedding_dim = self.embeddings.shape[1]
        self.index = AnnoyIndex(embedding_dim, metric=self.metric)

        if self.verbose:
            logger.info(
                f"Building Annoy index with {self.n_trees} trees for {len(self.embeddings)} embeddings"
            )
            iterator = tqdm(
                enumerate(self.embeddings),
                total=len(self.embeddings),
                desc="Adding embeddings to index",
            )
        else:
            iterator = enumerate(self.embeddings)

        for i, embedding_vector in iterator:
            self.index.add_item(i, embedding_vector)

        if self.verbose:
            logger.info("Building index trees...")
        self.index.build(self.n_trees)

        if self.verbose:
            logger.info("Vector index built successfully")

        return self.index

    def get_pairwise_similarities(
        self, thresh: float = 0.7, top_k: int = 100, inplace: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Find pairwise similarities between documents above a threshold.

        Uses the Annoy index to efficiently find nearest neighbors for each document,
        then calculates exact similarities and filters by threshold.

        Args:
            thresh: Minimum similarity threshold (0.0 to 1.0)
            top_k: Maximum number of neighbors to check per document
            inplace: If True, store results in self.neighbor_data. If False, return DataFrame.

        Returns:
            DataFrame of similarities if inplace=False, otherwise None

        Raises:
            ValueError: If embeddings or index haven't been built yet

        Note:
            Similarity is calculated as (1 - angular_distance) for angular metric.
            Results include columns: source_idx, target_idx, similarity, source_name, target_name
        """
        if self.embeddings is None or self.index is None:
            raise ValueError(
                "Embeddings or index not found. Please run embed_documents() and build_vector_index() first."
            )

        if self.verbose:
            logger.info(
                f"Finding pairwise similarities with threshold {thresh}, checking top {top_k} neighbors"
            )

        results = []

        # Use progress bar if verbose
        if self.verbose:
            iterator = tqdm(range(len(self.embeddings)), desc="Finding similarities")
        else:
            iterator = range(len(self.embeddings))

        for idx_source in iterator:
            neighbors = self.index.get_nns_by_item(
                idx_source, top_k, include_distances=True
            )

            for idx_target, dist in zip(*neighbors):
                similarity = 1 - dist  # Convert angular distance to similarity

                if idx_source != idx_target and similarity >= thresh:
                    result_dict = {
                        "source_idx": idx_source,
                        "target_idx": idx_target,
                        "similarity": similarity,
                        "source_name": self.docs[idx_source],
                        "target_name": self.docs[idx_target],
                    }
                    results.append(result_dict)

        self.neighbor_data = pd.DataFrame(results)

        if self.verbose:
            logger.info(
                f"Found {len(self.neighbor_data)} similarity pairs above threshold {thresh}"
            )

        if not inplace:
            return self.neighbor_data
        return None

    def build_graph(self) -> nx.Graph:
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
            The graph is stored in self.graph and also returned.
        """
        if self.neighbor_data is None:
            raise ValueError(
                "Neighbor data not found. Please run get_pairwise_similarities() first."
            )

        if self.verbose:
            logger.info(
                f"Building graph from {len(self.neighbor_data)} similarity edges"
            )

        # Create graph from edge list
        if len(self.neighbor_data) > 0:
            graph = nx.from_pandas_edgelist(
                self.neighbor_data,
                source="source_idx",
                target="target_idx",
                edge_attr="similarity",
            )
        else:
            # No similarities found, create empty graph
            graph = nx.Graph()

        # Create weight mapping
        weight_dict = {}
        if self.weights is not None:
            weight_dict = dict(zip(range(len(self.docs)), self.weights))

        # Update nodes with names and weights
        for node in graph.nodes:
            graph.nodes[node]["name"] = self.docs[node]
            graph.nodes[node]["weight"] = weight_dict.get(node, 1.0)  # Default to 1.0

        # Add isolated nodes (documents with no similarities above threshold)
        for i in range(len(self.docs)):
            if i not in graph.nodes:
                graph.add_node(i, name=self.docs[i], weight=weight_dict.get(i, 1.0))

        self.graph = graph

        if self.verbose:
            num_components = nx.number_connected_components(graph)
            logger.info(
                f"Built graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, {num_components} components"
            )

        return graph

    def get_deduplication_mapping(self) -> Dict[int, int]:
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
        if self.graph is None:
            raise ValueError("Graph not found. Please run build_graph() first.")

        components = list(nx.connected_components(self.graph))

        if self.verbose:
            logger.info(
                f"Found {len(components)} connected components for deduplication"
            )

        mapping = {}
        for component in components:
            if len(component) > 1:  # Only process components with multiple nodes
                subgraph = self.graph.subgraph(component)
                heaviest_node = max(
                    subgraph.nodes, key=lambda n: subgraph.nodes[n]["weight"]
                )

                # Map all other nodes in component to the heaviest node
                for node in subgraph.nodes:
                    if (
                        node != heaviest_node
                    ):  # Fixed bug: was comparing subgraph.nodes[node] != node
                        mapping[node] = heaviest_node

                if self.verbose and len(component) > 1:
                    component_docs = [self.docs[i] for i in component]
                    representative_doc = self.docs[heaviest_node]
                    logger.info(
                        f"Component of {len(component)} docs represented by: '{representative_doc[:50]}...'"
                    )

        if self.verbose:
            logger.info(
                f"Created mapping for {len(mapping)} documents to be deduplicated"
            )

        return mapping

    def deduplicate_documents(
        self, thresh: float = 0.7, top_k: int = 100
    ) -> Dict[str, Union[Dict[int, int], List[str], pd.DataFrame, nx.Graph]]:
        """
        Convenience method to run the full deduplication pipeline.

        Args:
            thresh: Similarity threshold for connecting documents
            top_k: Maximum neighbors to check per document

        Returns:
            Dictionary containing:
            - 'mapping': Dict[int, int] - Index mapping for deduplication
            - 'representatives': List[str] - List of representative documents
            - 'similarities': pd.DataFrame - Pairwise similarity data
            - 'graph': nx.Graph - The similarity graph
            - 'stats': Dict - Statistics about the deduplication
        """
        if self.verbose:
            logger.info("Starting full deduplication pipeline")

        # Run the pipeline
        self.embed_documents()
        self.build_vector_index()
        self.get_pairwise_similarities(thresh=thresh, top_k=top_k)
        self.build_graph()
        mapping = self.get_deduplication_mapping()

        # Get representatives (documents not in mapping)
        mapped_indices = set(mapping.keys())
        representative_indices = [
            i for i in range(len(self.docs)) if i not in mapped_indices
        ]
        representatives = [self.docs[i] for i in representative_indices]

        # Calculate statistics
        original_count = len(self.docs)
        deduplicated_count = len(representatives)
        reduction_ratio = (
            (original_count - deduplicated_count) / original_count
            if original_count > 0
            else 0
        )

        stats = {
            "original_count": original_count,
            "deduplicated_count": deduplicated_count,
            "duplicates_found": len(mapping),
            "reduction_ratio": reduction_ratio,
            "similarity_pairs": (
                len(self.neighbor_data) if self.neighbor_data is not None else 0
            ),
            "connected_components": (
                nx.number_connected_components(self.graph) if self.graph else 0
            ),
        }

        if self.verbose:
            logger.info(
                f"Deduplication complete: {original_count} -> {deduplicated_count} documents ({reduction_ratio:.1%} reduction)"
            )

        return {
            "mapping": mapping,
            "representatives": representatives,
            "similarities": self.neighbor_data,
            "graph": self.graph,
            "stats": stats,
        }

    def get_duplicate_groups(self) -> List[List[str]]:
        """
        Get groups of duplicate documents as lists of document text.

        Returns:
            List of groups, where each group contains the text of similar documents.
            Groups with only one document (no duplicates) are excluded.

        Raises:
            ValueError: If graph hasn't been built yet
        """
        if self.graph is None:
            raise ValueError("Graph not found. Please run build_graph() first.")

        components = list(nx.connected_components(self.graph))
        groups = []

        for component in components:
            if len(component) > 1:  # Only include groups with duplicates
                group_docs = [self.docs[i] for i in sorted(component)]
                groups.append(group_docs)

        # Sort by group size (largest first)
        groups.sort(key=len, reverse=True)

        if self.verbose:
            logger.info(f"Found {len(groups)} duplicate groups")

        return groups
