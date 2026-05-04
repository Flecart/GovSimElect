#!/usr/bin/env python3
"""
Parser for taxonomy.md that extracts the hierarchical structure
and identifies all leaf nodes for prompt generation.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import re


@dataclass
class TaxonomyNode:
    """Represents a node in the taxonomy tree."""
    name: str
    level: int
    parent: Optional['TaxonomyNode'] = None
    children: List['TaxonomyNode'] = field(default_factory=list)
    notes: Optional[str] = None  # Any notes in brackets or parentheses
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children) == 0
    
    def get_path(self) -> List[str]:
        """Get the full path from root to this node."""
        path = []
        node = self
        while node:
            path.insert(0, node.name)
            node = node.parent
        return path
    
    def get_path_string(self, separator: str = " > ") -> str:
        """Get the full path as a string."""
        return separator.join(self.get_path())


class TaxonomyParser:
    """Parser for the taxonomy.md file."""
    
    def __init__(self, file_path: str = "taxonomy.md"):
        self.file_path = file_path
        self.root_nodes: List[TaxonomyNode] = []
    
    def parse(self) -> List[TaxonomyNode]:
        """Parse the taxonomy file and return root nodes."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip the header line (starts with #)
        lines = [line for line in lines if not line.strip().startswith('#')]
        
        # Stack to keep track of parent nodes at each level
        stack: List[TaxonomyNode] = []
        
        for line in lines:
            if not line.strip():
                continue
            
            # Calculate indentation level (number of spaces before the dash)
            stripped = line.lstrip()
            if not stripped.startswith('-'):
                continue
            
            level = (len(line) - len(stripped)) // 2  # Assuming 2 spaces per level
            
            # Extract node name and notes
            content = stripped[1:].strip()  # Remove the dash
            node_name, notes = self._extract_notes(content)
            
            # Create new node
            node = TaxonomyNode(name=node_name, level=level, notes=notes)
            
            # Find parent in stack
            while stack and stack[-1].level >= level:
                stack.pop()
            
            if stack:
                parent = stack[-1]
                node.parent = parent
                parent.children.append(node)
            else:
                self.root_nodes.append(node)
            
            stack.append(node)
        
        return self.root_nodes
    
    def _extract_notes(self, content: str) -> tuple[str, Optional[str]]:
        """Extract notes from brackets or parentheses."""
        # Look for notes in brackets or parentheses at the end
        # Pattern: text [note] or text (note)
        bracket_match = re.search(r'\[([^\]]+)\]$', content)
        paren_match = re.search(r'\(([^)]+)\)$', content)
        
        if bracket_match:
            note = bracket_match.group(1)
            name = content[:bracket_match.start()].strip()
            return name, note
        elif paren_match:
            note = paren_match.group(1)
            name = content[:paren_match.start()].strip()
            return name, note
        else:
            return content, None
    
    def get_all_nodes(self, nodes: Optional[List[TaxonomyNode]] = None) -> List[TaxonomyNode]:
        """Get all nodes in the taxonomy tree (recursive)."""
        if nodes is None:
            nodes = self.root_nodes
        
        all_nodes = []
        for node in nodes:
            all_nodes.append(node)
            all_nodes.extend(self.get_all_nodes(node.children))
        return all_nodes
    
    def get_leaf_nodes(self) -> List[TaxonomyNode]:
        """Get all leaf nodes (nodes with no children)."""
        all_nodes = self.get_all_nodes()
        return [node for node in all_nodes if node.is_leaf()]
    
    def print_tree(self, nodes: Optional[List[TaxonomyNode]] = None, indent: int = 0):
        """Print the taxonomy tree structure."""
        if nodes is None:
            nodes = self.root_nodes
        
        for node in nodes:
            prefix = "  " * indent
            note_str = f" [{node.notes}]" if node.notes else ""
            leaf_marker = " (LEAF)" if node.is_leaf() else ""
            print(f"{prefix}- {node.name}{note_str}{leaf_marker}")
            if node.children:
                self.print_tree(node.children, indent + 1)


def main():
    """Example usage of the parser."""
    parser = TaxonomyParser("assets/taxonomy/taxonomy.md")
    parser.parse()
    
    print("=" * 60)
    print("Full Taxonomy Tree:")
    print("=" * 60)
    parser.print_tree()
    
    print("\n" + "=" * 60)
    print("All Leaf Nodes:")
    print("=" * 60)
    leaf_nodes = parser.get_leaf_nodes()
    for i, leaf in enumerate(leaf_nodes, 1):
        print(f"{i}. {leaf.get_path_string()}")
        if leaf.notes:
            print(f"   Notes: {leaf.notes}")
    
    print(f"\nTotal leaf nodes: {len(leaf_nodes)}")


if __name__ == "__main__":
    main()

