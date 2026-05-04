'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "ICD10 indexing utilities: lightweight in-memory lookup and optional SQLite FTS backend."
'''

from __future__ import annotations

## Standard library imports
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

## Third-party imports
import pandas as pd

## Internal imports
from src.utils.logging_utils import get_logger
from src.core.errors import (
    log_and_raise_missing_file,
)


## ============================================================
## LOGGER
## ============================================================
logger = get_logger("index_icd10", log_file="index_icd10.log")

## ============================================================
## IN-MEMORY INDEX
## ============================================================
class ICD10MemoryIndex:
    """
        Lightweight in-memory ICD10 lookup index

        Responsibilities:
            - Fast exact lookup by code
            - Simple keyword-based description search
    """

    ## ------------------------------------------------------------
    ## CONSTRUCTOR
    ## ------------------------------------------------------------
    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
            Initialize in-memory index from DataFrame

            Expected columns:
                - code
                - description

            Args:
                dataframe: Structured ICD10 DataFrame
        """

        ## Store internal dictionary for fast lookup
        self._by_code: Dict[str, str] = {}

        ## Normalize and load entries
        for _, row in dataframe.iterrows():
            code = str(row["code"]).strip()
            description = str(row["description"]).strip()

            if code:
                self._by_code[code] = description

        logger.info("Loaded %d ICD10 codes into memory index", len(self._by_code))

    ## ------------------------------------------------------------
    ## LOOKUP METHODS
    ## ------------------------------------------------------------
    def get_description(self, code: str) -> Optional[str]:
        """
            Retrieve description by code

            Args:
                code: ICD10 code

            Returns:
                Description or None
        """
        return self._by_code.get(code)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str]]:
        """
            Basic keyword search over descriptions

            Args:
                query: Search string
                top_k: Number of results to return

            Returns:
                List of (code, description)
        """

        query_lower = query.lower()

        ## Naive keyword filtering
        matches = [
            (code, desc)
            for code, desc in self._by_code.items()
            if query_lower in desc.lower()
        ]

        return matches[:top_k]

    def size(self) -> int:
        """
            Return number of indexed codes

            Returns:
                Number of codes
        """
        
        return len(self._by_code)

## ============================================================
## SQLITE FTS INDEX (OPTIONAL)
## ============================================================
class ICD10SQLiteIndex:
    """
        SQLite FTS-based ICD10 index

        Responsibilities:
            - Full-text search over ICD10 descriptions
            - Persistent storage
    """

    ## ------------------------------------------------------------
    ## CONSTRUCTOR
    ## ------------------------------------------------------------
    def __init__(self, db_path: str | Path) -> None:
        """
            Initialize SQLite index

            Args:
                db_path: Path to SQLite database
        """
        
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)

        ## Ensure FTS table exists
        self._initialize()

    ## ------------------------------------------------------------
    ## TABLE INITIALIZATION
    ## ------------------------------------------------------------
    def _initialize(self) -> None:
        """
            Create FTS table if not exists
        """
        
        cursor = self._conn.cursor()

        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS icd10_fts
            USING fts5(code, description);
            """
        )

        self._conn.commit()

    ## ------------------------------------------------------------
    ## BUILD INDEX
    ## ------------------------------------------------------------
    def build_from_dataframe(self, dataframe: pd.DataFrame) -> None:
        """
            Populate SQLite FTS index from DataFrame

            Args:
                dataframe: ICD10 structured data
        """

        cursor = self._conn.cursor()

        ## Clear existing entries
        cursor.execute("DELETE FROM icd10_fts;")

        ## Insert rows
        for _, row in dataframe.iterrows():
            cursor.execute(
                "INSERT INTO icd10_fts (code, description) VALUES (?, ?);",
                (row["code"], row["description"]),
            )

        self._conn.commit()

        logger.info("SQLite ICD10 index built with %d entries", len(dataframe))

    ## ------------------------------------------------------------
    ## SEARCH
    ## ------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str]]:
        """
            Full-text search using SQLite FTS

            Args:
                query: Search query
                top_k: Number of results

            Returns:
                List of (code, description)
        """

        cursor = self._conn.cursor()

        cursor.execute(
            """
            SELECT code, description
            FROM icd10_fts
            WHERE icd10_fts MATCH ?
            LIMIT ?;
            """,
            (query, top_k),
        )

        results = cursor.fetchall()

        return results

    ## ------------------------------------------------------------
    ## CLOSE CONNECTION
    ## ------------------------------------------------------------
    def close(self) -> None:
        """
            Close SQLite connection
        """
        
        self._conn.close()

## ============================================================
## HELPER FUNCTION
## ============================================================
def load_icd10_csv(csv_path: str | Path) -> pd.DataFrame:
    """
        Load structured ICD10 CSV

        Args:
            csv_path: Path to ICD10 CSV file

        Returns:
            Pandas DataFrame
    """

    path = Path(csv_path)

    if not path.exists():
        log_and_raise_missing_file(path)

    logger.info("Loading ICD10 CSV from %s", path)

    return pd.read_csv(path)