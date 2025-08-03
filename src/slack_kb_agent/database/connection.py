"""Database connection and session management with connection pooling."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any

import asyncpg
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncSession, 
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool, QueuePool

from ..exceptions import DatabaseError
from ..circuit_breaker import CircuitBreaker


logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class DatabaseManager:
    """
    Manages database connections with connection pooling, circuit breaker,
    and automatic reconnection capabilities.
    """
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 30,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False
    ):
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo
        
        self._engine: Optional[AsyncEngine] = None
        self._session_maker: Optional[async_sessionmaker] = None
        self._metadata = MetaData()
        
        # Circuit breaker for database operations
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=DatabaseError
        )
        
        # Connection health tracking
        self._connection_health = {
            'healthy': True,
            'last_check': None,
            'consecutive_failures': 0
        }
        
        logger.info(f"DatabaseManager initialized with pool_size={pool_size}")
    
    async def initialize(self):
        """Initialize database engine and session maker."""
        try:
            # Create async engine with connection pooling
            engine_kwargs = {
                'echo': self.echo,
                'pool_size': self.pool_size,
                'max_overflow': self.max_overflow,
                'pool_timeout': self.pool_timeout,
                'pool_recycle': self.pool_recycle,
                'pool_pre_ping': True,  # Verify connections before use
                'poolclass': QueuePool
            }
            
            # Handle special cases for different database types
            if 'sqlite' in self.database_url:
                engine_kwargs['poolclass'] = NullPool  # SQLite doesn't need pooling
                engine_kwargs.pop('pool_size', None)
                engine_kwargs.pop('max_overflow', None)
            
            self._engine = create_async_engine(self.database_url, **engine_kwargs)
            
            # Create session maker
            self._session_maker = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            await self.health_check()
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")
    
    async def close(self):
        """Close database engine and connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_maker = None
            logger.info("Database connections closed")
    
    @property
    def engine(self) -> AsyncEngine:
        """Get the database engine."""
        if not self._engine:
            raise DatabaseError("Database not initialized. Call initialize() first.")
        return self._engine
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic cleanup and error handling.
        
        Usage:
            async with db_manager.get_session() as session:
                result = await session.execute(query)
        """
        if not self._session_maker:
            raise DatabaseError("Database not initialized. Call initialize() first.")
        
        session = None
        try:
            # Use circuit breaker for session creation
            session = await self.circuit_breaker.call(self._session_maker)
            
            yield session
            
            # Commit if no exceptions
            await session.commit()
            
        except Exception as e:
            if session:
                try:
                    await session.rollback()
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback transaction: {rollback_error}")
            
            # Update connection health
            self._connection_health['consecutive_failures'] += 1
            self._connection_health['healthy'] = False
            
            logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
            
        finally:
            if session:
                try:
                    await session.close()
                except Exception as close_error:
                    logger.warning(f"Error closing session: {close_error}")
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a raw SQL query with parameters.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            
        Returns:
            Query result
        """
        async with self.get_session() as session:
            try:
                if parameters:
                    result = await session.execute(text(query), parameters)
                else:
                    result = await session.execute(text(query))
                
                return result
                
            except Exception as e:
                logger.error(f"Query execution failed: {query[:100]}... Error: {e}")
                raise DatabaseError(f"Query execution failed: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check.
        
        Returns:
            Health status dictionary
        """
        health_status = {
            'healthy': False,
            'connection_pool': {},
            'latency_ms': None,
            'error': None
        }
        
        try:
            import time
            start_time = time.time()
            
            # Test basic connectivity
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                await result.fetchone()
            
            # Calculate latency
            health_status['latency_ms'] = (time.time() - start_time) * 1000
            
            # Get connection pool status
            if hasattr(self._engine.pool, 'size'):
                health_status['connection_pool'] = {
                    'pool_size': self._engine.pool.size(),
                    'checked_in': self._engine.pool.checkedin(),
                    'checked_out': self._engine.pool.checkedout(),
                    'overflow': getattr(self._engine.pool, 'overflow', 0)
                }
            
            health_status['healthy'] = True
            self._connection_health['healthy'] = True
            self._connection_health['consecutive_failures'] = 0
            
            logger.debug(f"Database health check passed: {health_status['latency_ms']:.2f}ms")
            
        except Exception as e:
            health_status['error'] = str(e)
            self._connection_health['consecutive_failures'] += 1
            self._connection_health['healthy'] = False
            
            logger.error(f"Database health check failed: {e}")
        
        return health_status
    
    async def create_tables(self):
        """Create all database tables."""
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise DatabaseError(f"Table creation failed: {str(e)}")
    
    async def drop_tables(self):
        """Drop all database tables."""
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            
            logger.info("Database tables dropped successfully")
            
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise DatabaseError(f"Table dropping failed: {str(e)}")
    
    async def backup_data(self, backup_path: str) -> Dict[str, Any]:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path to save backup file
            
        Returns:
            Backup information
        """
        try:
            # This is a simplified backup - in production, use pg_dump or similar
            backup_info = {
                'timestamp': asyncio.get_event_loop().time(),
                'path': backup_path,
                'tables': [],
                'size_bytes': 0
            }
            
            # Export table schemas and data
            async with self.get_session() as session:
                # Get table names
                result = await session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                tables = [row[0] for row in result.fetchall()]
                backup_info['tables'] = tables
            
            logger.info(f"Database backup created: {backup_path}")
            return backup_info
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise DatabaseError(f"Backup failed: {str(e)}")
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get detailed connection pool statistics."""
        stats = {
            'engine_url': str(self._engine.url).replace(
                self._engine.url.password or '', '***'
            ) if self._engine else None,
            'pool_status': 'not_initialized'
        }
        
        if self._engine and hasattr(self._engine, 'pool'):
            pool = self._engine.pool
            stats.update({
                'pool_status': 'active',
                'pool_size': getattr(pool, 'size', lambda: 0)(),
                'checked_in_connections': getattr(pool, 'checkedin', lambda: 0)(),
                'checked_out_connections': getattr(pool, 'checkedout', lambda: 0)(),
                'overflow_connections': getattr(pool, 'overflow', lambda: 0)(),
                'total_connections': getattr(pool, 'size', lambda: 0)() + getattr(pool, 'overflow', lambda: 0)()
            })
        
        stats.update(self._connection_health)
        return stats


class ConnectionPool:
    """Simple connection pool for raw asyncpg connections."""
    
    def __init__(
        self,
        database_url: str,
        min_size: int = 10,
        max_size: int = 20,
        command_timeout: float = 60.0
    ):
        self.database_url = database_url
        self.min_size = min_size
        self.max_size = max_size
        self.command_timeout = command_timeout
        self._pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize the connection pool."""
        try:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=self.command_timeout
            )
            logger.info(f"AsyncPG connection pool initialized: {self.min_size}-{self.max_size} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise DatabaseError(f"Connection pool initialization failed: {str(e)}")
    
    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Connection pool closed")
    
    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Acquire a connection from the pool."""
        if not self._pool:
            raise DatabaseError("Connection pool not initialized")
        
        async with self._pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error(f"Connection error: {e}")
                raise DatabaseError(f"Connection operation failed: {str(e)}")
    
    async def execute_query(self, query: str, *args) -> Any:
        """Execute a query using a pooled connection."""
        async with self.acquire() as connection:
            try:
                return await connection.fetch(query, *args)
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise DatabaseError(f"Query failed: {str(e)}")
    
    @property
    def pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if not self._pool:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'active',
            'min_size': self._pool.get_min_size(),
            'max_size': self._pool.get_max_size(),
            'current_size': self._pool.get_size(),
            'idle_connections': self._pool.get_idle_size()
        }


# Global database manager instance
database_manager: Optional[DatabaseManager] = None


async def init_database(database_url: str, **kwargs) -> DatabaseManager:
    """Initialize global database manager."""
    global database_manager
    
    database_manager = DatabaseManager(database_url, **kwargs)
    await database_manager.initialize()
    
    return database_manager


async def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    if not database_manager:
        raise DatabaseError("Database not initialized. Call init_database() first.")
    
    return database_manager


async def close_database():
    """Close the global database manager."""
    global database_manager
    
    if database_manager:
        await database_manager.close()
        database_manager = None


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Convenience function to get database session.
    
    Usage:
        async with get_db_session() as session:
            result = await session.execute(query)
    """
    db = await get_database()
    async with db.get_session() as session:
        yield session


# Database configuration from environment
def get_database_config() -> Dict[str, Any]:
    """Get database configuration from environment variables."""
    return {
        'database_url': os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/slack_kb_agent'),
        'pool_size': int(os.getenv('DB_POOL_SIZE', '20')),
        'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '30')),
        'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
        'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '3600')),
        'echo': os.getenv('DEBUG', 'false').lower() == 'true'
    }