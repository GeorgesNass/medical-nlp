'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Label definitions and per-label configuration (thresholds, descriptions, keyword hints)."
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.core.config import CONFIG, LABELS, LABEL_DESCRIPTIONS, LABEL_KEYWORD_HINTS


## -----------------------------
## Data structures
## -----------------------------
@dataclass(frozen=True)
class LabelDefinition:
    """
        Label definition and configuration

        Attributes:
            name: Label name (e.g., crh, cro, cra)
            description: Human-friendly description
            threshold: Cosine similarity threshold for binary decision
            keyword_hints: Optional keyword hints (weak signals for EDA/debug)
    """

    name: str
    description: str
    threshold: float
    keyword_hints: List[str]


## -----------------------------
## Builders
## -----------------------------
def build_label_definitions() -> Dict[str, LabelDefinition]:
    """
        Build label definitions from CONFIG and global label mappings

        Returns:
            Dict of label name -> LabelDefinition
    """

    ## Build a consistent definition per label
    defs: Dict[str, LabelDefinition] = {}

    for label in LABELS:
        ## Thresholds come from CONFIG (env overrides already applied)
        threshold = float(CONFIG.similarity.thresholds.get(label, 0.55))

        ## Descriptions and keyword hints are optional but helpful
        description = str(LABEL_DESCRIPTIONS.get(label, label))
        keyword_hints = list(LABEL_KEYWORD_HINTS.get(label, []))

        defs[label] = LabelDefinition(
            name=label,
            description=description,
            threshold=threshold,
            keyword_hints=keyword_hints,
        )

    return defs


def get_label_names() -> Tuple[str, ...]:
    """
        Return label names as a tuple (stable order)

        Returns:
            Tuple of label names
    """

    return tuple(LABELS)
