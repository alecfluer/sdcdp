"""
Provides the Social Distance Configuration model with Degree Preservation (SDC-DP) for sampling networks. 
Includes utilities for generating correlated degree sequences and computing connection probabilities using 
the Social Distance Attachment (SDA) method generalized to undirected multiplex networks.

See docs/ for documentation and full mathematical details.
"""

from . import sda
from . import sdcdp
from . import seq

__all__ = ["sda", "sdcdp", "seq"]

__version__ = "0.1.0"
__author__ = "Alec Fluer"
__url__ = "https://github.com/alecfluer/sdcdp"
__license__ = "MIT"