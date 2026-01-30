"""
Video module for Digital Witness.

Provides video loading, clip extraction, and annotated clip generation.
"""
from .loader import VideoLoader
from .clip_extractor import ClipExtractor
from .annotated_clip_generator import AnnotatedClipGenerator, AnnotatedClip

__all__ = [
    'VideoLoader',
    'ClipExtractor',
    'AnnotatedClipGenerator',
    'AnnotatedClip'
]
