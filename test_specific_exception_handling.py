#!/usr/bin/env python3
"""
Test to verify that broad exception handlers have been replaced with specific ones.
This test helps ensure better error handling and debugging capabilities.
"""

import ast
import os
from pathlib import Path
from typing import List, Tuple, Dict


class ExceptionHandlerAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze exception handlers."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.broad_exceptions = []
        self.specific_exceptions = []
        self.acceptable_broad_exceptions = []
    
    def visit_ExceptHandler(self, node):
        line_no = node.lineno
        
        # Check if it's a broad Exception handler
        if (node.type and 
            isinstance(node.type, ast.Name) and 
            node.type.id == 'Exception'):
            
            # Get context around the exception handler
            context = self._get_exception_context(node)
            
            # Check if this is an acceptable use case
            if self._is_acceptable_broad_exception(context, node):
                self.acceptable_broad_exceptions.append((line_no, context))
            else:
                self.broad_exceptions.append((line_no, context))
        
        elif node.type:
            # This is a specific exception
            if isinstance(node.type, ast.Name):
                exc_type = node.type.id
            elif isinstance(node.type, ast.Attribute):
                exc_type = f"{node.type.value.id}.{node.type.attr}"
            elif isinstance(node.type, ast.Tuple):
                exc_types = []
                for elt in node.type.elts:
                    if isinstance(elt, ast.Name):
                        exc_types.append(elt.id)
                    elif isinstance(elt, ast.Attribute):
                        exc_types.append(f"{elt.value.id}.{elt.attr}")
                exc_type = f"({', '.join(exc_types)})"
            else:
                exc_type = "unknown"
            
            context = self._get_exception_context(node)
            self.specific_exceptions.append((line_no, exc_type, context))
        
        self.generic_visit(node)
    
    def _get_exception_context(self, node):
        """Get context information about the exception handler."""
        # Look for comments or specific patterns in the handler body
        if node.body:
            first_stmt = node.body[0]
            if (isinstance(first_stmt, ast.Expr) and 
                isinstance(first_stmt.value, ast.Constant) and 
                isinstance(first_stmt.value.value, str)):
                return first_stmt.value.value
        return "generic_handler"
    
    def _is_acceptable_broad_exception(self, context: str, node) -> bool:
        """
        Determine if a broad Exception handler is acceptable.
        Some cases where broad handlers might be necessary:
        - CLI tools with final error handling
        - Plugin loading systems
        - Network/IO operations with multiple failure modes
        - Fallback error handling in user interfaces
        """
        acceptable_patterns = [
            "defensive",      # Explicitly marked as defensive
            "fallback",       # Fallback error handling
            "cli",           # CLI error handling
            "plugin",        # Plugin loading
            "final",         # Final error boundary
            "ui",           # UI error boundary
        ]
        
        return any(pattern in context.lower() for pattern in acceptable_patterns)


def analyze_exception_handlers(src_dir: Path) -> Dict[str, ExceptionHandlerAnalyzer]:
    """Analyze exception handlers in all Python files."""
    analyzers = {}
    
    for python_file in src_dir.rglob("*.py"):
        if python_file.name.startswith('test_'):
            continue  # Skip test files for now
        
        try:
            with open(python_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            analyzer = ExceptionHandlerAnalyzer(str(python_file))
            analyzer.visit(tree)
            
            if (analyzer.broad_exceptions or 
                analyzer.specific_exceptions or 
                analyzer.acceptable_broad_exceptions):
                analyzers[str(python_file)] = analyzer
                
        except Exception as e:
            print(f"Error analyzing {python_file}: {e}")
    
    return analyzers


def main():
    """Main test function."""
    src_dir = Path("src/slack_kb_agent")
    
    if not src_dir.exists():
        print(f"❌ Source directory {src_dir} does not exist")
        return False
    
    print("🔍 Analyzing exception handlers...")
    analyzers = analyze_exception_handlers(src_dir)
    
    # Count totals
    total_broad = sum(len(a.broad_exceptions) for a in analyzers.values())
    total_specific = sum(len(a.specific_exceptions) for a in analyzers.values())
    total_acceptable = sum(len(a.acceptable_broad_exceptions) for a in analyzers.values())
    
    print(f"\n📊 Exception Handler Analysis Results:")
    print(f"   • Specific exception handlers: {total_specific}")
    print(f"   • Acceptable broad handlers: {total_acceptable}")  
    print(f"   • Problematic broad handlers: {total_broad}")
    
    # Report problematic broad exception handlers
    if total_broad > 0:
        print(f"\n❌ Found {total_broad} broad exception handlers that should be made specific:")
        for filepath, analyzer in analyzers.items():
            if analyzer.broad_exceptions:
                rel_path = filepath.replace(str(Path.cwd()) + "/", "")
                print(f"\n   📄 {rel_path}:")
                for line_no, context in analyzer.broad_exceptions:
                    print(f"      • Line {line_no}: {context}")
    
    # Report acceptable broad handlers (for documentation)
    if total_acceptable > 0:
        print(f"\n✅ Found {total_acceptable} acceptable broad exception handlers:")
        for filepath, analyzer in analyzers.items():
            if analyzer.acceptable_broad_exceptions:
                rel_path = filepath.replace(str(Path.cwd()) + "/", "")
                print(f"   📄 {rel_path}: {len(analyzer.acceptable_broad_exceptions)} handlers")
    
    # Success criteria
    success = total_broad == 0
    
    if success:
        print(f"\n🎉 SUCCESS: All exception handlers are appropriately specific!")
    else:
        print(f"\n⚠️  NEEDS WORK: {total_broad} broad exception handlers need to be made specific")
        print("   Consider replacing 'except Exception:' with specific exception types like:")
        print("   • except (ValueError, TypeError, KeyError):")
        print("   • except requests.RequestException:")
        print("   • except json.JSONDecodeError:")
        print("   • except IOError:")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)