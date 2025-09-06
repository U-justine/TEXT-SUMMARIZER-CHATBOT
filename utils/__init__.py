"""
Utilities package for PDF Text Summarizer & ChatBot

This package contains utility modules for text processing and model management.
"""

from .text_processor import TextProcessor
from .model_utils import ModelManager

__all__ = ['TextProcessor', 'ModelManager']

# Version information
__version__ = '1.0.0'
__author__ = 'PDF Summarizer Team'
__description__ = 'Utility modules for PDF text processing and ML model management'
