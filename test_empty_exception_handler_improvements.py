#!/usr/bin/env python3
"""
Test to verify that empty exception handlers have been improved with proper
error handling, logging, or recovery mechanisms.
"""

import ast
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set
import tempfile
from unittest.mock import patch


class EmptyExceptionHandlerAnalyzer(ast.NodeVisitor):
    """AST visitor to find empty exception handlers."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.empty_handlers = []
        self.improved_handlers = []
        self.acceptable_handlers = []
    
    def visit_ExceptHandler(self, node):
        line_no = node.lineno
        
        # Check if exception handler body only contains pass
        if self._is_empty_handler(node):
            context = self._get_handler_context(node)
            
            # Check if this is an acceptable empty handler (e.g., imports)
            if self._is_acceptable_empty_handler(context, node):
                self.acceptable_handlers.append((line_no, context))
            else:
                self.empty_handlers.append((line_no, context))
        elif self._has_proper_error_handling(node):
            context = self._get_handler_context(node)
            self.improved_handlers.append((line_no, context))
        
        self.generic_visit(node)
    
    def _is_empty_handler(self, node) -> bool:
        """Check if exception handler only contains pass statement."""
        if not node.body:
            return True
        
        # Single pass statement
        if (len(node.body) == 1 and 
            isinstance(node.body[0], ast.Pass)):
            return True
        
        # Pass with optional comment
        if (len(node.body) == 2 and 
            isinstance(node.body[0], ast.Pass) and
            isinstance(node.body[1], ast.Expr) and
            isinstance(node.body[1].value, ast.Constant)):
            return True
        
        return False
    
    def _has_proper_error_handling(self, node) -> bool:
        """Check if handler has proper error handling (logging, recovery, etc.)."""
        if not node.body:
            return False
        
        for stmt in node.body:
            # Has logging calls
            if self._contains_logging_call(stmt):
                return True
            # Has error recovery logic
            if self._contains_recovery_logic(stmt):
                return True
            # Has re-raising or new exception
            if isinstance(stmt, (ast.Raise, ast.Return)):
                return True
        
        return False
    
    def _contains_logging_call(self, node) -> bool:
        """Check if node contains logging calls."""
        if isinstance(node, ast.Expr):
            call = node.value
            if isinstance(call, ast.Call):
                if isinstance(call.func, ast.Attribute):
                    # logger.error(), logger.warning(), etc.
                    if (isinstance(call.func.value, ast.Name) and
                        call.func.value.id in ['logger', 'logging']):
                        return True
                    # print() calls for error output
                    if (isinstance(call.func.value, ast.Name) and
                        call.func.value.id == 'print'):
                        return True
        return False
    
    def _contains_recovery_logic(self, node) -> bool:
        """Check if node contains error recovery logic."""
        # Assignment statements (setting defaults, flags, etc.)
        if isinstance(node, ast.Assign):
            return True
        # If statements for conditional recovery
        if isinstance(node, ast.If):
            return True
        # Function calls that might be recovery actions
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return True
        return False
    
    def _get_handler_context(self, node) -> str:
        """Get context information about the exception handler."""
        if node.type:
            if isinstance(node.type, ast.Name):
                return node.type.id
            elif isinstance(node.type, ast.Tuple):
                types = []
                for elt in node.type.elts:
                    if isinstance(elt, ast.Name):
                        types.append(elt.id)
                return f"({', '.join(types)})"
        return "generic"
    
    def _is_acceptable_empty_handler(self, context: str, node) -> bool:
        """Check if empty handler is acceptable (e.g., import error handling)."""
        acceptable_patterns = [
            "ImportError",   # Graceful import handling
            "KeyboardInterrupt",  # User interruption
            "SystemExit",    # Clean exit handling
        ]
        return any(pattern in context for pattern in acceptable_patterns)


def analyze_empty_exception_handlers(src_dir: Path) -> Dict[str, EmptyExceptionHandlerAnalyzer]:
    """Analyze empty exception handlers in all Python files."""
    analyzers = {}
    
    for python_file in src_dir.rglob("*.py"):
        if python_file.name.startswith('test_'):
            continue  # Skip test files for now
        
        try:
            with open(python_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            analyzer = EmptyExceptionHandlerAnalyzer(str(python_file))
            analyzer.visit(tree)
            
            if (analyzer.empty_handlers or 
                analyzer.improved_handlers or 
                analyzer.acceptable_handlers):
                analyzers[str(python_file)] = analyzer
                
        except Exception as e:
            print(f"Error analyzing {python_file}: {e}")
    
    return analyzers


def create_test_file_with_empty_handler() -> Path:
    """Create a test file with empty exception handler for testing."""
    test_content = '''
def test_function():
    try:
        risky_operation()
    except ValueError:
        pass  # Empty handler - should be improved
        
def test_function_with_logging():
    try:
        another_operation()
    except TypeError as e:
        logger.error(f"Type error occurred: {e}")
        
def acceptable_empty_handler():
    try:
        import optional_module
    except ImportError:
        pass  # Acceptable - optional import
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_content)
        return Path(f.name)


def test_empty_handler_analyzer():
    """Test the empty exception handler analyzer."""
    test_file = create_test_file_with_empty_handler()
    
    try:
        analyzer = EmptyExceptionHandlerAnalyzer(str(test_file))
        
        with open(test_file, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
        analyzer.visit(tree)
        
        # Should find one empty handler (ValueError)
        assert len(analyzer.empty_handlers) == 1, f"Expected 1 empty handler, found {len(analyzer.empty_handlers)}"
        assert analyzer.empty_handlers[0][1] == "ValueError"
        
        # Should find one improved handler (TypeError with logging)
        assert len(analyzer.improved_handlers) == 1, f"Expected 1 improved handler, found {len(analyzer.improved_handlers)}"
        
        # Should find one acceptable empty handler (ImportError)
        assert len(analyzer.acceptable_handlers) == 1, f"Expected 1 acceptable handler, found {len(analyzer.acceptable_handlers)}"
        assert "ImportError" in analyzer.acceptable_handlers[0][1]
        
        print("✅ Empty handler analyzer test passed")
        
    finally:
        test_file.unlink()


def main():
    """Main test function."""
    print("🧪 Testing Empty Exception Handler Analysis...")
    
    # Test the analyzer with synthetic data
    test_empty_handler_analyzer()
    
    # Analyze the actual codebase
    src_dir = Path("src/slack_kb_agent")
    
    if not src_dir.exists():
        print(f"❌ Source directory {src_dir} does not exist")
        return False
    
    print("🔍 Analyzing exception handlers in codebase...")
    analyzers = analyze_empty_exception_handlers(src_dir)
    
    # Count totals
    total_empty = sum(len(a.empty_handlers) for a in analyzers.values())
    total_improved = sum(len(a.improved_handlers) for a in analyzers.values())
    total_acceptable = sum(len(a.acceptable_handlers) for a in analyzers.values())
    
    print(f"\n📊 Empty Exception Handler Analysis Results:")
    print(f"   • Proper error handlers: {total_improved}")
    print(f"   • Acceptable empty handlers: {total_acceptable}")  
    print(f"   • Empty handlers needing improvement: {total_empty}")
    
    # Report empty handlers that need improvement
    if total_empty > 0:
        print(f"\n⚠️  Found {total_empty} empty exception handlers that need improvement:")
        for filepath, analyzer in analyzers.items():
            if analyzer.empty_handlers:
                rel_path = filepath.replace(str(Path.cwd()) + "/", "")
                print(f"\n   📄 {rel_path}:")
                for line_no, context in analyzer.empty_handlers:
                    print(f"      • Line {line_no}: except {context}: pass")
    
    # Report improved handlers (for verification)
    if total_improved > 0:
        print(f"\n✅ Found {total_improved} properly handled exception handlers:")
        for filepath, analyzer in analyzers.items():
            if analyzer.improved_handlers:
                rel_path = filepath.replace(str(Path.cwd()) + "/", "")
                print(f"   📄 {rel_path}: {len(analyzer.improved_handlers)} proper handlers")
    
    # Success criteria: prefer fewer empty handlers
    success = total_empty == 0
    
    if success:
        print(f"\n🎉 SUCCESS: All exception handlers have proper error handling!")
    else:
        print(f"\n🔧 IMPROVEMENT NEEDED: {total_empty} empty handlers should be enhanced with:")
        print("   • Logging: logger.error(f'Error description: {e}')")
        print("   • Recovery: set default values, retry logic, or cleanup")
        print("   • Documentation: explain why the exception can be safely ignored")
        print("   • Re-raising: raise more specific exception if needed")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)