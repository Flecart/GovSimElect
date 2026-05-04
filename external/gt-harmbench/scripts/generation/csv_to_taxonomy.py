#!/usr/bin/env python3
"""
Transform data/mit.csv to assets/taxonomy/taxonomy.md format.

This script reads the CSV file and extracts:
- Risk category (main categories)
- Risk subcategory (subcategories)
- Description (descriptions for categories/subcategories)

And generates a markdown taxonomy file in the format:
# AI Risk Taxonomy
- Category Name
    - Subcategory Name: Description text
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def read_csv_data(csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Read CSV and organize data by category and subcategory.
    
    Returns:
        Dictionary mapping category -> {subcategory: description}
        Empty string as subcategory key means it's a category-level description
    """
    data = defaultdict(dict)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            category_level = row.get('Category level', '').strip()
            risk_category = row.get('Risk category', '').strip()
            risk_subcategory = row.get('Risk subcategory', '').strip()
            description = row.get('Description', '').strip()
            
            # Only process Risk Category and Risk Sub-Category rows
            if category_level not in ['Risk Category', 'Risk Sub-Category']:
                continue
            
            # Skip if no category name
            if not risk_category:
                continue
            
            # Clean up description (remove extra quotes if present)
            if description:
                # Remove surrounding quotes if they wrap the entire description
                if description.startswith('"') and description.endswith('"'):
                    description = description[1:-1]
                # Replace double quotes with single quotes
                description = description.replace('""', '"')
            
            if category_level == 'Risk Category':
                # This is a main category
                # Store category-level description with empty subcategory key
                if description:
                    data[risk_category][''] = description
            elif category_level == 'Risk Sub-Category':
                # This is a subcategory
                if risk_subcategory:
                    data[risk_category][risk_subcategory] = description
                else:
                    # Subcategory level but no subcategory name - treat as category description
                    if description:
                        data[risk_category][''] = description
    
    return data


def generate_markdown(data: Dict[str, Dict[str, str]]) -> str:
    """
    Generate markdown taxonomy from organized data.
    
    Args:
        data: Dictionary mapping category -> {subcategory: description}
    
    Returns:
        Markdown string
    """
    lines = ["# AI Risk Taxonomy"]
    
    # Sort categories alphabetically
    sorted_categories = sorted(data.keys())
    
    for category in sorted_categories:
        subcategories = data[category]
        
        if "Not-Suitable-for-Work" in category:
            continue
        # Add main category
        category_desc = subcategories.get('', '')
        lines.append(f"- {category}: {category_desc}")
        
        # Get category-level description if exists
        
        # Get all subcategories (excluding the empty key)
        subcat_items = [(k, v) for k, v in subcategories.items() if k]
        
        # Sort subcategories alphabetically
        subcat_items.sort(key=lambda x: x[0])
        
        # If there are subcategories, add them
        if subcat_items:
            for subcat, desc in subcat_items:
                if desc:
                    lines.append(f"    - {subcat}: {desc}")
                else:
                    lines.append(f"    - {subcat}")
        # If no subcategories but there's a category description, add it
        elif category_desc:
            # Indent the description under the category
            lines.append(f"    ")
        
        # Add blank line between categories
        lines.append("")
    
    return "\n".join(lines)


def main():
    """Main function to transform CSV to markdown taxonomy."""
    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    csv_path = os.path.join(project_root, "data", "mit.csv")
    output_path = os.path.join(project_root, "assets", "taxonomy", "taxonomy.md")
    
    print(f"Reading CSV from: {csv_path}")
    data = read_csv_data(csv_path)
    
    print(f"Found {len(data)} categories")
    total_subcategories = sum(len([k for k in v.keys() if k]) for v in data.values())
    print(f"Found {total_subcategories} subcategories")
    
    print(f"Generating markdown...")
    markdown = generate_markdown(data)
    
    print(f"Writing to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print("Done!")


if __name__ == "__main__":
    main()

