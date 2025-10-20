"""Semnet: Semantic Network Deduplication

A Python package for building semantic networks using embeddings and graph clustering
to perform intelligent deduplication of text data.
"""

__version__ = "0.1.0"
__author__ = "Ian Goodrich"
__email__ = "ian@igdr.ch"

from .semnet import SemanticNetwork

__all__ = [
    "SemanticNetwork",
]
