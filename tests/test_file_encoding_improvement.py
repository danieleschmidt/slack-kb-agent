"""Test file encoding error handling improvement."""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
from src.slack_kb_agent.ingestion import FileIngester


class TestFileEncodingImprovement(unittest.TestCase):
    """Test the file encoding error handling improvement."""

    def test_encoding_errors_use_replace_strategy(self):
        """Test that encoding errors use replace strategy instead of ignore."""
        # Create a temporary file with invalid UTF-8 bytes
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as f:
            # Write some valid UTF-8 followed by invalid bytes
            f.write(b'Valid text here \xff\xfe Invalid bytes \x80\x81 More valid text')
            temp_file = f.name
        
        try:
            ingester = FileIngester()
            file_path = Path(temp_file)
            
            # Mock logger to capture warnings
            with patch('src.slack_kb_agent.ingestion.logger') as mock_logger:
                doc = ingester.ingest_file(file_path)
                
                # Verify document was created
                self.assertIsNotNone(doc, "Document should be created despite encoding issues")
                
                # Check that content contains replacement characters
                self.assertIn('�', doc.content, "Content should contain replacement characters for invalid bytes")
                
                # Verify that valid text is preserved
                self.assertIn('Valid text here', doc.content, "Valid text should be preserved")
                self.assertIn('More valid text', doc.content, "Valid text should be preserved")
                
                # Check that a warning was logged about encoding issues
                mock_logger.warning.assert_called()
                warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
                encoding_warning_found = any('encoding' in call.lower() for call in warning_calls)
                self.assertTrue(encoding_warning_found, "Should log warning about encoding issues")
                
        finally:
            # Clean up
            os.unlink(temp_file)

    def test_valid_utf8_files_work_normally(self):
        """Test that valid UTF-8 files continue to work normally."""
        # Create a temporary file with valid UTF-8 content
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write('This is valid UTF-8 content with unicode: café, naïve, résumé')
            temp_file = f.name
        
        try:
            ingester = FileIngester()
            file_path = Path(temp_file)
            
            # Mock logger to verify no warnings
            with patch('src.slack_kb_agent.ingestion.logger') as mock_logger:
                doc = ingester.ingest_file(file_path)
                
                # Verify document was created
                self.assertIsNotNone(doc, "Document should be created for valid UTF-8")
                
                # Check that content is preserved correctly
                self.assertIn('café', doc.content, "Unicode characters should be preserved")
                self.assertIn('naïve', doc.content, "Unicode characters should be preserved")
                self.assertIn('résumé', doc.content, "Unicode characters should be preserved")
                
                # Should not contain replacement characters
                self.assertNotIn('�', doc.content, "Valid UTF-8 should not contain replacement characters")
                
                # Should not log encoding warnings for valid files
                if mock_logger.warning.called:
                    warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
                    encoding_warning_found = any('encoding' in call.lower() for call in warning_calls)
                    self.assertFalse(encoding_warning_found, "Should not log encoding warnings for valid UTF-8")
                
        finally:
            # Clean up
            os.unlink(temp_file)

    def test_empty_file_handling(self):
        """Test that empty files are handled correctly."""
        # Create an empty temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_file = f.name
        
        try:
            ingester = FileIngester()
            file_path = Path(temp_file)
            
            doc = ingester.ingest_file(file_path)
            
            # Empty files should return None
            self.assertIsNone(doc, "Empty files should return None document")
                
        finally:
            # Clean up
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()