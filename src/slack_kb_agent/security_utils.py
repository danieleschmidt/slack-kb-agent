#!/usr/bin/env python3
"""
Security utilities for credential masking and sensitive data protection.

This module provides utilities to mask sensitive information like database
credentials, API keys, and other secrets in logs, monitoring, and error messages.
"""

import re
from typing import Optional, Dict, Any


def mask_database_url(url: Optional[str]) -> str:
    """
    Safely mask credentials in database URLs for logging and monitoring.
    
    This function masks passwords and other sensitive information in database
    connection strings while preserving enough information for debugging.
    
    Args:
        url: Database URL that may contain credentials
        
    Returns:
        Masked URL with credentials replaced by *** markers
        
    Examples:
        >>> mask_database_url("postgresql://user:password@localhost:5432/db")
        "postgresql://user:***@localhost:5432/db"
        
        >>> mask_database_url("mysql://admin:secret@db.example.com/mydb") 
        "mysql://admin:***@db.example.com/mydb"
        
        >>> mask_database_url("postgresql://localhost/db")
        "postgresql://localhost/db"
    """
    if not url or not isinstance(url, str):
        return str(url) if url is not None else "None"
    
    try:
        # Check if this looks like a database URL
        if '://' not in url:
            return url  # Not a URL format
        
        # Parse URL parts
        scheme_netloc = url.split('://', 1)
        if len(scheme_netloc) != 2:
            return url
            
        scheme, rest = scheme_netloc
        
        # Check if there are credentials (username:password@host format)
        if '@' not in rest:
            return url  # No credentials to mask
        
        # Split credentials and host info
        creds_host = rest.split('@', 1)
        if len(creds_host) != 2:
            return url
            
        credentials, host_db = creds_host
        
        # Mask the password part while preserving username
        if ':' in credentials:
            username, password = credentials.split(':', 1)
            if password:  # Only mask if password is not empty
                masked_creds = f"{username}:***"
            else:
                masked_creds = credentials  # No password to mask
        else:
            # Only username, no password
            masked_creds = credentials
        
        return f"{scheme}://{masked_creds}@{host_db}"
        
    except Exception:
        # If URL parsing fails for any reason, return a safe generic indicator
        return f"<masked_database_url_parse_error>"


def mask_connection_string(connection_str: Optional[str]) -> str:
    """
    Mask sensitive information in various connection string formats.
    
    This handles different connection string formats beyond just URLs,
    including key-value pair formats used by some database drivers.
    
    Args:
        connection_str: Connection string that may contain sensitive data
        
    Returns:
        Masked connection string with sensitive data replaced
    """
    if not connection_str or not isinstance(connection_str, str):
        return str(connection_str) if connection_str is not None else "None"
    
    try:
        # Handle URL format first
        if '://' in connection_str:
            return mask_database_url(connection_str)
        
        # Handle key-value format (e.g., "host=localhost password=secret")
        if '=' in connection_str:
            # Split by spaces but preserve the structure
            parts = connection_str.split(' ')
            masked_parts = []
            
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    # Mask password-like keys
                    if key.lower() in ['password', 'pwd', 'pass', 'secret', 'key']:
                        masked_parts.append(f"{key}=***")
                    else:
                        masked_parts.append(part)
                else:
                    masked_parts.append(part)
            
            return ' '.join(masked_parts)
        
        # If we can't parse it, assume it might be sensitive and mask it
        return "<masked_connection_string>"
        
    except Exception:
        return "<masked_connection_string_error>"


def mask_sensitive_dict(data: Dict[str, Any], 
                       sensitive_keys: Optional[set] = None) -> Dict[str, Any]:
    """
    Recursively mask sensitive values in dictionaries for safe logging.
    
    Args:
        data: Dictionary that may contain sensitive data
        sensitive_keys: Set of keys to consider sensitive (default includes common ones)
        
    Returns:
        New dictionary with sensitive values masked
    """
    if sensitive_keys is None:
        sensitive_keys = {
            'password', 'pwd', 'pass', 'secret', 'key', 'token', 'auth',
            'credential', 'api_key', 'database_url', 'connection_string',
            'private_key', 'access_token', 'refresh_token'
        }
    
    if not isinstance(data, dict):
        return data
    
    masked_data = {}
    
    for key, value in data.items():
        key_lower = key.lower()
        
        if key_lower in sensitive_keys:
            # Mask the entire value for sensitive keys
            if isinstance(value, str) and value:
                # For URLs, use our URL masking
                if '://' in value:
                    masked_data[key] = mask_database_url(value)
                else:
                    masked_data[key] = "***"
            else:
                masked_data[key] = "***"
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            masked_data[key] = mask_sensitive_dict(value, sensitive_keys)
        elif isinstance(value, str) and any(pattern in value for pattern in ['://', 'password=', 'secret=']):
            # Check if string value might contain connection info
            masked_data[key] = mask_connection_string(value)
        else:
            # Keep non-sensitive values as-is
            masked_data[key] = value
    
    return masked_data


def get_safe_repr(obj: Any, mask_attrs: Optional[set] = None) -> str:
    """
    Get a safe string representation of an object with sensitive attributes masked.
    
    Args:
        obj: Object to represent
        mask_attrs: Set of attribute names to mask (default includes common sensitive ones)
        
    Returns:
        Safe string representation with sensitive data masked
    """
    if mask_attrs is None:
        mask_attrs = {
            'password', 'database_url', 'connection_string', 'api_key', 
            'secret', 'token', 'credential'
        }
    
    try:
        class_name = obj.__class__.__name__
        
        # Get relevant attributes
        attrs = []
        for attr_name in dir(obj):
            if not attr_name.startswith('_') and not callable(getattr(obj, attr_name, None)):
                try:
                    value = getattr(obj, attr_name)
                    
                    # Mask sensitive attributes
                    if attr_name.lower() in mask_attrs:
                        if isinstance(value, str) and '://' in value:
                            attrs.append(f"{attr_name}={mask_database_url(value)}")
                        else:
                            attrs.append(f"{attr_name}=***")
                    else:
                        # For non-sensitive attributes, use a simple representation
                        if isinstance(value, str):
                            attrs.append(f"{attr_name}='{value[:50]}{'...' if len(value) > 50 else ''}'")
                        else:
                            attrs.append(f"{attr_name}={type(value).__name__}")
                            
                except Exception:
                    # Skip attributes that can't be accessed
                    continue
        
        if attrs:
            return f"{class_name}({', '.join(attrs[:5])}{'...' if len(attrs) > 5 else ''})"
        else:
            return f"{class_name}()"
            
    except Exception:
        # Fallback to basic representation
        return f"<{type(obj).__name__} object>"


# Pre-compiled regex patterns for performance
_URL_PASSWORD_PATTERN = re.compile(r'(://[^:]+:)[^@]+(@)', re.IGNORECASE)
_CONNECTION_PASSWORD_PATTERN = re.compile(r'(password|pwd|pass|secret|key)\s*=\s*[^\s;]+', re.IGNORECASE)


def quick_mask_credentials(text: str) -> str:
    """
    Quick regex-based credential masking for performance-critical contexts.
    
    This is a lighter-weight alternative to mask_database_url for cases where
    performance is critical and the input format is predictable.
    
    Args:
        text: Text that may contain credentials
        
    Returns:
        Text with credentials masked using regex replacement
    """
    if not text:
        return text
    
    # Mask passwords in URLs
    text = _URL_PASSWORD_PATTERN.sub(r'\1***\2', text)
    
    # Mask password-like key-value pairs
    text = _CONNECTION_PASSWORD_PATTERN.sub(r'\1=***', text)
    
    return text