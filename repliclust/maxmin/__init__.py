"""
This module provides functionality for implementing data set archetypes
based on max-min sampling.
"""

from .archetype import MaxMinArchetype
from .covariance import MaxMinCovarianceSampler
from .groupsizes import MaxMinGroupSizeSampler
from repliclust.optimized import ArchetypeOptimized

__all__ = [
    'base',
    'overlap',
    'maxmin',
    'distributions',
    'distortion',
    'viz',
    'Archetype',
    'ArchetypeOptimized'
]