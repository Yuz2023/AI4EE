#!/usr/bin/env python3
"""
Fix character encoding issues in the streamlined files
"""

import os
import sys

def fix_file_encoding(filepath):
    """Fix common encoding issues in a file"""
    
    replacements = {
        'Ã—': '×',
        'â†'': '→',
        'â†"': '↓',
        'âœ"': '✓',
        'âœ—': '✗',
        'ðŸš€': '🚀',
        'ðŸ"Š': '📊',
        'ðŸ§ª': '🧪',
        'ðŸ"‹': '📋',
        'ðŸ"¦': '📦',
        'ðŸŽ¯': '🎯',
        'ðŸ"§': '🔧',
        'ðŸ"': '📁',
        'ðŸŽ›ï¸': '🎛️',
        'ðŸ"ˆ': '📈',
        'ðŸ›': '🐛',
        'ðŸ"': '📝',
        'ðŸ"„': '📄',
        'ðŸ¤': '🤝',
        'ðŸ†˜': '🆘',
        'ðŸŽ‰': '🎉',
        'â­': '⭐',
        'âœ¨': '✨',
        'ðŸ›ï¸': '🛠️',
        'ðŸ³': '🐳',
        'ðŸ§ª': '🧪',
        'â"œâ"€â"€': '├──',
        'â""â"€â"€': '└──',
        'â"‚': '│',
    }
    
    try:
        # Read file with UTF-8 encoding
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Apply replacements
        original = content
        for bad, good in replacements.items():
            content = content.replace(bad, good)
        
        # Write back if changed
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed encoding in: {filepath}")
            return True
        else:
            print(f"No encoding issues in: {filepath}")
            return False
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Fix encoding in all Python and text files"""
    
    files_to_fix = [
        'model_manager.py',
        'bulk_tester.py', 
        'llm_cli.py',
        'config.yaml',
        'Makefile',
        'quick_setup.sh',
        'README.md'
    ]
    
    fixed_count = 0
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            if fix_file_encoding(filepath):
                fixed_count += 1
        else:
            print(f"Warning: {filepath} not found")
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()