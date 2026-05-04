'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized metadata builder combining DIM, time and analysis group extraction."
'''

from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.metadata.dim_extractions import extract_dim_metadata
from src.metadata.time_extractions import extract_time_metadata
from src.metadata.analysis_group import detect_analysis_group
from src.utils.logging_utils import get_logger
from src.utils.utils import normalize_clinical_text

logger = get_logger(__name__)

## ============================================================
## METADATA BUILDER
## ============================================================
def build_metadata(
    text: str,
    source_file: str | Path,
) -> Dict[str, str]:
    """
        Build full metadata dictionary from raw TXT content

        High-level workflow:
            1) Extract DIM metadata (gender, dates_dob)
            2) Extract time metadata (sampling_time, dates_edition)
            3) Detect analysis group
            4) Attach source file name
            5) Return consolidated metadata dict

        Args:
            text: Raw TXT content
            source_file: Source file name or path

        Returns:
            Dictionary with:
                - file
                - gender
                - dates_dob
                - sampling_time
                - dates_edition
                - analysis_group
    """

    ## Ensure text is not None
    content = text or ""

    normalized_content = normalize_clinical_text(content)
    
    ## Extract DIM-related metadata
    #dim_meta = extract_dim_metadata(content)
    dim_meta = extract_dim_metadata(normalized_content)
    
    ## Extract time-related metadata
    #time_meta = extract_time_metadata(content)
    time_meta = extract_time_metadata(normalized_content)

    ## Detect analysis group
    #analysis_group = detect_analysis_group(content)
    analysis_group = detect_analysis_group(normalized_content)

    ## Normalize file name
    file_name = Path(source_file).name

    ## Consolidate metadata
    metadata = {
        "file": file_name,
        "gender": dim_meta.get("gender", ""),
        "dates_dob": dim_meta.get("dates_dob", ""),
        "sampling_time": time_meta.get("sampling_time", ""),
        "dates_edition": time_meta.get("dates_edition", ""),
        "analysis_group": analysis_group,
        "char_length": len(normalized_content),
        "token_count": len(normalized_content.split()),
        
    }

    return metadata