'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "ICD10 taxonomy utilities: hierarchical operations, parent-child resolution and roll-up logic."
'''

from __future__ import annotations

## Standard library imports
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

## ============================================================
## DATA STRUCTURES
## ============================================================
@dataclass
class ICD10Node:
    """
        ICD10 taxonomy node

        Attributes:
            code: ICD10 code
            description: Human-readable label
            parent: Parent ICD10 code (if any)
            children: List of child ICD10 codes
    """

    code: str
    description: str
    parent: Optional[str]
    children: List[str]

## ============================================================
## TAXONOMY CLASS
## ============================================================
class ICD10Taxonomy:
    """
        In-memory ICD10 hierarchical structure

        Responsibilities:
            - Store ICD10 nodes
            - Provide parent lookup
            - Provide children lookup
            - Support roll-up operations
    """

    ## ------------------------------------------------------------
    ## CONSTRUCTOR
    ## ------------------------------------------------------------
    def __init__(self) -> None:
        """
            Initialize empty taxonomy
        """
        ## Dictionary: code -> ICD10Node
        self._nodes: Dict[str, ICD10Node] = {}

    ## ------------------------------------------------------------
    ## NODE MANAGEMENT
    ## ------------------------------------------------------------
    def add_node(
        self,
        code: str,
        description: str,
        parent: Optional[str] = None,
    ) -> None:
        """
            Add a node to taxonomy

            Args:
                code: ICD10 code
                description: Human-readable description
                parent: Optional parent ICD10 code
        """

        ## Avoid duplicates
        if code in self._nodes:
            return

        ## Create node
        node = ICD10Node(
            code=code,
            description=description,
            parent=parent,
            children=[],
        )

        self._nodes[code] = node

        ## Register child under parent if exists
        if parent and parent in self._nodes:
            self._nodes[parent].children.append(code)

    ## ------------------------------------------------------------
    ## LOOKUPS
    ## ------------------------------------------------------------
    def get_node(self, code: str) -> Optional[ICD10Node]:
        """
            Retrieve node by code

            Args:
                code: ICD10 code

            Returns:
                ICD10Node or None
        """
        
        return self._nodes.get(code)

    def get_parent(self, code: str) -> Optional[str]:
        """
            Get parent code

            Args:
                code: ICD10 code

            Returns:
                Parent code or None
        """
        
        node = self.get_node(code)
        return node.parent if node else None

    def get_children(self, code: str) -> List[str]:
        """
            Get children codes

            Args:
                code: ICD10 code

            Returns:
                List of child ICD10 codes
        """
        
        node = self.get_node(code)       
        return node.children if node else []

    ## ------------------------------------------------------------
    ## HIERARCHY OPERATIONS
    ## ------------------------------------------------------------
    def get_ancestors(self, code: str) -> List[str]:
        """
            Retrieve all ancestors of a code

            Args:
                code: ICD10 code

            Returns:
                Ordered list of ancestor codes (closest first)
        """

        ancestors: List[str] = []
        current = code

        ## Walk up the tree
        while True:
            parent = self.get_parent(current)
            if not parent:
                break

            ancestors.append(parent)
            current = parent

        return ancestors

    def get_descendants(self, code: str) -> List[str]:
        """
            Retrieve all descendants of a code

            Args:
                code: ICD10 code

            Returns:
                List of descendant ICD10 codes
        """

        descendants: Set[str] = set()
        stack = [code]

        ## Depth-first traversal
        while stack:
            current = stack.pop()
            children = self.get_children(current)

            for child in children:
                if child not in descendants:
                    descendants.add(child)
                    stack.append(child)

        return list(descendants)

    def roll_up_to_level(self, code: str, level_depth: int) -> Optional[str]:
        """
            Roll-up ICD10 code to a specific hierarchy depth

            Args:
                code: ICD10 code
                level_depth: Number of parent levels to move up

            Returns:
                Rolled-up ICD10 code or None if not possible
        """

        current = code

        for _ in range(level_depth):
            parent = self.get_parent(current)
            if not parent:
                return None
            current = parent

        return current

    ## ------------------------------------------------------------
    ## UTILITIES
    ## ------------------------------------------------------------
    def contains(self, code: str) -> bool:
        """
            Check if taxonomy contains code

            Args:
                code: ICD10 code

            Returns:
                True if exists
        """
        
        return code in self._nodes

    def size(self) -> int:
        """
            Return number of nodes

            Returns:
                Number of ICD10 codes
        """
        
        return len(self._nodes)