"""
Event Intelligence Module
Unified SEC + news event intelligence for supervised ensemble trading system.
"""

from .event_quant_hybrid import EventQuantHybrid, create_default_config
from .event_ingestion import EventIngestion
from .event_classifier import EventClassifier
from .event_encoder import EventEncoder
from .event_archive import EventArchive

__all__ = [
    'EventQuantHybrid',
    'EventIngestion',
    'EventClassifier',
    'EventEncoder',
    'EventArchive',
    'create_default_config'
]

__version__ = '1.1.0'
