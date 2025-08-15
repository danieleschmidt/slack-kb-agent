"""Secure password hashing utilities using bcrypt."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not available - password hashing disabled")


class PasswordHasher:
    """Secure password hashing using bcrypt with salt."""

    def __init__(self, cost: int = 12):
        """Initialize password hasher.
        
        Args:
            cost: bcrypt cost parameter (higher = more secure but slower)
                  Recommended: 12-14 for production, 4-6 for testing
        """
        if not BCRYPT_AVAILABLE:
            raise ImportError("bcrypt library is required for password hashing")

        if cost < 4 or cost > 18:
            raise ValueError("Cost parameter must be between 4 and 18")

        self.cost = cost
        logger.debug(f"PasswordHasher initialized with cost={cost}")

    def hash_password(self, password: str) -> str:
        """Hash a password with a random salt.
        
        Args:
            password: Plain text password to hash
            
        Returns:
            Hashed password with salt (bcrypt format)
            
        Raises:
            ValueError: If password is too long (>72 bytes after encoding)
        """
        if not isinstance(password, str):
            raise TypeError("Password must be a string")

        # Encode password to bytes
        try:
            password_bytes = password.encode('utf-8')
        except UnicodeEncodeError as e:
            raise ValueError(f"Password contains invalid characters: {e}")

        # bcrypt truncates passwords at 72 bytes - warn about this
        if len(password_bytes) > 72:
            logger.warning(
                f"Password is {len(password_bytes)} bytes, bcrypt will truncate to 72 bytes"
            )

        try:
            # Generate salt and hash password
            salt = bcrypt.gensalt(rounds=self.cost)
            hashed = bcrypt.hashpw(password_bytes, salt)

            # Return as string for storage
            return hashed.decode('utf-8')

        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            raise ValueError(f"Failed to hash password: {e}")

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            password: Plain text password to verify
            hashed_password: Previously hashed password to verify against
            
        Returns:
            True if password matches, False otherwise
        """
        if not isinstance(password, str) or not isinstance(hashed_password, str):
            return False

        if not hashed_password:
            return False

        try:
            # Encode inputs to bytes
            password_bytes = password.encode('utf-8')
            hash_bytes = hashed_password.encode('utf-8')

            # Use bcrypt to verify password
            return bcrypt.checkpw(password_bytes, hash_bytes)

        except (ValueError, TypeError, UnicodeEncodeError) as e:
            # Log error but don't leak information
            logger.debug(f"Password verification failed: {e}")
            return False
        except Exception as e:
            # Unexpected error - log but return False
            logger.error(f"Unexpected error in password verification: {e}")
            return False

    def is_hash(self, value: str) -> bool:
        """Check if a string looks like a bcrypt hash.
        
        Args:
            value: String to check
            
        Returns:
            True if value appears to be a bcrypt hash
        """
        if not isinstance(value, str):
            return False

        # bcrypt hashes start with $2a$, $2b$, $2x$, or $2y$ and are ~60 chars
        return (
            len(value) >= 50 and
            value.startswith(('$2a$', '$2b$', '$2x$', '$2y$')) and
            value.count('$') >= 3
        )

    def needs_rehash(self, hashed_password: str, target_cost: Optional[int] = None) -> bool:
        """Check if a password hash needs to be rehashed (due to cost change).
        
        Args:
            hashed_password: Existing password hash
            target_cost: Target cost level (defaults to instance cost)
            
        Returns:
            True if hash should be rehashed with new parameters
        """
        if not self.is_hash(hashed_password):
            return True

        target_cost = target_cost or self.cost

        try:
            # Extract cost from hash
            parts = hashed_password.split('$')
            if len(parts) >= 3:
                current_cost = int(parts[2])
                return current_cost < target_cost
        except (ValueError, IndexError):
            # If we can't parse the hash, it should be rehashed
            return True

        return False


# Convenience functions for backward compatibility
def hash_password(password: str, cost: int = 12) -> str:
    """Hash a password using default hasher.
    
    Args:
        password: Plain text password to hash
        cost: bcrypt cost parameter
        
    Returns:
        Hashed password
    """
    hasher = PasswordHasher(cost=cost)
    return hasher.hash_password(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password using default hasher.
    
    Args:
        password: Plain text password to verify
        hashed_password: Previously hashed password
        
    Returns:
        True if password matches
    """
    hasher = PasswordHasher()
    return hasher.verify_password(password, hashed_password)


def is_bcrypt_hash(value: str) -> bool:
    """Check if a string is a bcrypt hash.
    
    Args:
        value: String to check
        
    Returns:
        True if value appears to be a bcrypt hash
    """
    hasher = PasswordHasher()
    return hasher.is_hash(value)
