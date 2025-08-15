"""Base repository with common CRUD operations and query patterns."""

import logging
from abc import ABC
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sqlalchemy import and_, asc, delete, desc, func, or_, select, update
from sqlalchemy.exc import IntegrityError

from ..database.connection import get_db_session
from ..database.models import Base
from ..exceptions import RepositoryError

logger = logging.getLogger(__name__)

# Generic type for model classes
ModelType = TypeVar('ModelType', bound=Base)


class BaseRepository(Generic[ModelType], ABC):
    """
    Base repository providing common CRUD operations and query patterns.
    Implements the Repository pattern for data access abstraction.
    """

    def __init__(self, model_class: Type[ModelType]):
        self.model_class = model_class
        self.table_name = model_class.__tablename__
        logger.debug(f"Repository initialized for {self.table_name}")

    async def create(self, **kwargs) -> ModelType:
        """
        Create a new record.
        
        Args:
            **kwargs: Field values for the new record
            
        Returns:
            Created model instance
            
        Raises:
            RepositoryError: If creation fails
        """
        try:
            async with get_db_session() as session:
                instance = self.model_class(**kwargs)
                session.add(instance)
                await session.flush()  # Get the ID without committing
                await session.refresh(instance)  # Refresh to get computed fields

                logger.debug(f"Created {self.table_name} record: {instance.id}")
                return instance

        except IntegrityError as e:
            logger.error(f"Integrity error creating {self.table_name}: {e}")
            raise RepositoryError(f"Failed to create {self.table_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating {self.table_name}: {e}")
            raise RepositoryError(f"Failed to create {self.table_name}: {str(e)}")

    async def get_by_id(self, record_id: str) -> Optional[ModelType]:
        """
        Get record by ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            Model instance or None if not found
        """
        try:
            async with get_db_session() as session:
                stmt = select(self.model_class).where(self.model_class.id == record_id)
                result = await session.execute(stmt)
                instance = result.scalar_one_or_none()

                if instance:
                    logger.debug(f"Found {self.table_name} record: {record_id}")

                return instance

        except Exception as e:
            logger.error(f"Error getting {self.table_name} by ID {record_id}: {e}")
            raise RepositoryError(f"Failed to get {self.table_name}: {str(e)}")

    async def get_by_field(self, field_name: str, value: Any) -> Optional[ModelType]:
        """
        Get record by specific field value.
        
        Args:
            field_name: Name of the field to search
            value: Value to search for
            
        Returns:
            Model instance or None if not found
        """
        try:
            async with get_db_session() as session:
                field = getattr(self.model_class, field_name)
                stmt = select(self.model_class).where(field == value)
                result = await session.execute(stmt)
                instance = result.scalar_one_or_none()

                if instance:
                    logger.debug(f"Found {self.table_name} record by {field_name}: {value}")

                return instance

        except AttributeError:
            raise RepositoryError(f"Field {field_name} does not exist on {self.table_name}")
        except Exception as e:
            logger.error(f"Error getting {self.table_name} by {field_name}: {e}")
            raise RepositoryError(f"Failed to get {self.table_name}: {str(e)}")

    async def get_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False
    ) -> List[ModelType]:
        """
        Get all records with optional pagination and ordering.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            order_by: Field name to order by
            order_desc: Whether to order in descending order
            
        Returns:
            List of model instances
        """
        try:
            async with get_db_session() as session:
                stmt = select(self.model_class)

                # Apply ordering
                if order_by:
                    field = getattr(self.model_class, order_by, None)
                    if field is not None:
                        if order_desc:
                            stmt = stmt.order_by(desc(field))
                        else:
                            stmt = stmt.order_by(asc(field))

                # Apply pagination
                if offset:
                    stmt = stmt.offset(offset)
                if limit:
                    stmt = stmt.limit(limit)

                result = await session.execute(stmt)
                instances = result.scalars().all()

                logger.debug(f"Retrieved {len(instances)} {self.table_name} records")
                return list(instances)

        except Exception as e:
            logger.error(f"Error getting all {self.table_name}: {e}")
            raise RepositoryError(f"Failed to get {self.table_name} records: {str(e)}")

    async def update(self, record_id: str, **kwargs) -> Optional[ModelType]:
        """
        Update record by ID.
        
        Args:
            record_id: Record ID
            **kwargs: Fields to update
            
        Returns:
            Updated model instance or None if not found
        """
        try:
            async with get_db_session() as session:
                # Add updated_at timestamp if the model has this field
                if hasattr(self.model_class, 'updated_at'):
                    kwargs['updated_at'] = datetime.utcnow()

                stmt = (
                    update(self.model_class)
                    .where(self.model_class.id == record_id)
                    .values(**kwargs)
                    .returning(self.model_class)
                )

                result = await session.execute(stmt)
                instance = result.scalar_one_or_none()

                if instance:
                    logger.debug(f"Updated {self.table_name} record: {record_id}")

                return instance

        except IntegrityError as e:
            logger.error(f"Integrity error updating {self.table_name}: {e}")
            raise RepositoryError(f"Failed to update {self.table_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error updating {self.table_name}: {e}")
            raise RepositoryError(f"Failed to update {self.table_name}: {str(e)}")

    async def delete(self, record_id: str) -> bool:
        """
        Delete record by ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            async with get_db_session() as session:
                stmt = delete(self.model_class).where(self.model_class.id == record_id)
                result = await session.execute(stmt)

                deleted = result.rowcount > 0
                if deleted:
                    logger.debug(f"Deleted {self.table_name} record: {record_id}")

                return deleted

        except Exception as e:
            logger.error(f"Error deleting {self.table_name}: {e}")
            raise RepositoryError(f"Failed to delete {self.table_name}: {str(e)}")

    async def exists(self, record_id: str) -> bool:
        """
        Check if record exists by ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            True if exists, False otherwise
        """
        try:
            async with get_db_session() as session:
                stmt = select(func.count()).where(self.model_class.id == record_id)
                result = await session.execute(stmt)
                count = result.scalar()

                return count > 0

        except Exception as e:
            logger.error(f"Error checking existence of {self.table_name}: {e}")
            raise RepositoryError(f"Failed to check {self.table_name} existence: {str(e)}")

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records with optional filters.
        
        Args:
            filters: Dictionary of field filters
            
        Returns:
            Number of matching records
        """
        try:
            async with get_db_session() as session:
                stmt = select(func.count(self.model_class.id))

                if filters:
                    conditions = []
                    for field_name, value in filters.items():
                        field = getattr(self.model_class, field_name, None)
                        if field is not None:
                            if isinstance(value, list):
                                conditions.append(field.in_(value))
                            else:
                                conditions.append(field == value)

                    if conditions:
                        stmt = stmt.where(and_(*conditions))

                result = await session.execute(stmt)
                count = result.scalar()

                logger.debug(f"Counted {count} {self.table_name} records")
                return count

        except Exception as e:
            logger.error(f"Error counting {self.table_name}: {e}")
            raise RepositoryError(f"Failed to count {self.table_name}: {str(e)}")

    async def find_by_filters(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False
    ) -> List[ModelType]:
        """
        Find records by multiple filters.
        
        Args:
            filters: Dictionary of field filters
            limit: Maximum number of records
            offset: Number of records to skip
            order_by: Field to order by
            order_desc: Descending order flag
            
        Returns:
            List of matching model instances
        """
        try:
            async with get_db_session() as session:
                stmt = select(self.model_class)

                # Apply filters
                if filters:
                    conditions = []
                    for field_name, value in filters.items():
                        field = getattr(self.model_class, field_name, None)
                        if field is not None:
                            if isinstance(value, list):
                                conditions.append(field.in_(value))
                            elif isinstance(value, dict):
                                # Handle range filters like {'gte': 100, 'lt': 200}
                                if 'gte' in value:
                                    conditions.append(field >= value['gte'])
                                if 'gt' in value:
                                    conditions.append(field > value['gt'])
                                if 'lte' in value:
                                    conditions.append(field <= value['lte'])
                                if 'lt' in value:
                                    conditions.append(field < value['lt'])
                                if 'like' in value:
                                    conditions.append(field.like(f"%{value['like']}%"))
                            else:
                                conditions.append(field == value)

                    if conditions:
                        stmt = stmt.where(and_(*conditions))

                # Apply ordering
                if order_by:
                    field = getattr(self.model_class, order_by, None)
                    if field is not None:
                        if order_desc:
                            stmt = stmt.order_by(desc(field))
                        else:
                            stmt = stmt.order_by(asc(field))

                # Apply pagination
                if offset:
                    stmt = stmt.offset(offset)
                if limit:
                    stmt = stmt.limit(limit)

                result = await session.execute(stmt)
                instances = result.scalars().all()

                logger.debug(f"Found {len(instances)} {self.table_name} records matching filters")
                return list(instances)

        except Exception as e:
            logger.error(f"Error finding {self.table_name} by filters: {e}")
            raise RepositoryError(f"Failed to find {self.table_name}: {str(e)}")

    async def bulk_create(self, records: List[Dict[str, Any]]) -> List[ModelType]:
        """
        Create multiple records in a single transaction.
        
        Args:
            records: List of dictionaries with record data
            
        Returns:
            List of created model instances
        """
        try:
            async with get_db_session() as session:
                instances = []
                for record_data in records:
                    instance = self.model_class(**record_data)
                    session.add(instance)
                    instances.append(instance)

                await session.flush()  # Get IDs without committing

                # Refresh all instances to get computed fields
                for instance in instances:
                    await session.refresh(instance)

                logger.debug(f"Bulk created {len(instances)} {self.table_name} records")
                return instances

        except IntegrityError as e:
            logger.error(f"Integrity error in bulk create for {self.table_name}: {e}")
            raise RepositoryError(f"Failed to bulk create {self.table_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in bulk create for {self.table_name}: {e}")
            raise RepositoryError(f"Failed to bulk create {self.table_name}: {str(e)}")

    async def bulk_update(self, updates: List[Dict[str, Any]]) -> int:
        """
        Update multiple records by ID.
        
        Args:
            updates: List of dictionaries with 'id' and update fields
            
        Returns:
            Number of updated records
        """
        try:
            async with get_db_session() as session:
                updated_count = 0

                for update_data in updates:
                    record_id = update_data.pop('id')

                    # Add updated_at timestamp if the model has this field
                    if hasattr(self.model_class, 'updated_at'):
                        update_data['updated_at'] = datetime.utcnow()

                    stmt = (
                        update(self.model_class)
                        .where(self.model_class.id == record_id)
                        .values(**update_data)
                    )

                    result = await session.execute(stmt)
                    updated_count += result.rowcount

                logger.debug(f"Bulk updated {updated_count} {self.table_name} records")
                return updated_count

        except Exception as e:
            logger.error(f"Error in bulk update for {self.table_name}: {e}")
            raise RepositoryError(f"Failed to bulk update {self.table_name}: {str(e)}")

    async def bulk_delete(self, record_ids: List[str]) -> int:
        """
        Delete multiple records by ID.
        
        Args:
            record_ids: List of record IDs to delete
            
        Returns:
            Number of deleted records
        """
        try:
            async with get_db_session() as session:
                stmt = delete(self.model_class).where(self.model_class.id.in_(record_ids))
                result = await session.execute(stmt)

                deleted_count = result.rowcount
                logger.debug(f"Bulk deleted {deleted_count} {self.table_name} records")
                return deleted_count

        except Exception as e:
            logger.error(f"Error in bulk delete for {self.table_name}: {e}")
            raise RepositoryError(f"Failed to bulk delete {self.table_name}: {str(e)}")

    async def search_text(
        self,
        search_term: str,
        search_fields: List[str],
        limit: Optional[int] = None
    ) -> List[ModelType]:
        """
        Search for records using text search across specified fields.
        
        Args:
            search_term: Text to search for
            search_fields: List of field names to search in
            limit: Maximum number of results
            
        Returns:
            List of matching model instances
        """
        try:
            async with get_db_session() as session:
                conditions = []

                for field_name in search_fields:
                    field = getattr(self.model_class, field_name, None)
                    if field is not None:
                        # Use ILIKE for case-insensitive search
                        conditions.append(field.ilike(f"%{search_term}%"))

                if not conditions:
                    return []

                stmt = select(self.model_class).where(or_(*conditions))

                if limit:
                    stmt = stmt.limit(limit)

                result = await session.execute(stmt)
                instances = result.scalars().all()

                logger.debug(f"Text search found {len(instances)} {self.table_name} records")
                return list(instances)

        except Exception as e:
            logger.error(f"Error in text search for {self.table_name}: {e}")
            raise RepositoryError(f"Failed to search {self.table_name}: {str(e)}")

    async def get_aggregated_stats(self, group_by: str, aggregate_field: str, aggregate_func: str = 'count') -> Dict[str, Any]:
        """
        Get aggregated statistics grouped by a field.
        
        Args:
            group_by: Field to group by
            aggregate_field: Field to aggregate
            aggregate_func: Aggregation function ('count', 'sum', 'avg', 'min', 'max')
            
        Returns:
            Dictionary with aggregated results
        """
        try:
            async with get_db_session() as session:
                group_field = getattr(self.model_class, group_by)
                agg_field = getattr(self.model_class, aggregate_field)

                # Choose aggregation function
                if aggregate_func == 'count':
                    agg_expr = func.count(agg_field)
                elif aggregate_func == 'sum':
                    agg_expr = func.sum(agg_field)
                elif aggregate_func == 'avg':
                    agg_expr = func.avg(agg_field)
                elif aggregate_func == 'min':
                    agg_expr = func.min(agg_field)
                elif aggregate_func == 'max':
                    agg_expr = func.max(agg_field)
                else:
                    raise ValueError(f"Unsupported aggregate function: {aggregate_func}")

                stmt = (
                    select(group_field, agg_expr)
                    .group_by(group_field)
                    .order_by(desc(agg_expr))
                )

                result = await session.execute(stmt)
                rows = result.fetchall()

                stats = {str(row[0]): row[1] for row in rows}

                logger.debug(f"Generated aggregated stats for {self.table_name}")
                return stats

        except Exception as e:
            logger.error(f"Error generating stats for {self.table_name}: {e}")
            raise RepositoryError(f"Failed to generate stats for {self.table_name}: {str(e)}")


class CacheableRepository(BaseRepository[ModelType]):
    """Repository with built-in caching capabilities."""

    def __init__(self, model_class: Type[ModelType], cache_manager=None):
        super().__init__(model_class)
        self.cache_manager = cache_manager
        self.cache_ttl = 3600  # 1 hour default

    def _get_cache_key(self, operation: str, *args) -> str:
        """Generate cache key for operation."""
        key_parts = [self.table_name, operation] + [str(arg) for arg in args]
        return ":".join(key_parts)

    async def get_by_id(self, record_id: str) -> Optional[ModelType]:
        """Get record by ID with caching."""
        if not self.cache_manager:
            return await super().get_by_id(record_id)

        cache_key = self._get_cache_key("get_by_id", record_id)
        cached_result = await self.cache_manager.get(cache_key)

        if cached_result is not None:
            return cached_result

        result = await super().get_by_id(record_id)
        if result:
            await self.cache_manager.set(cache_key, result, ttl=self.cache_ttl)

        return result

    async def invalidate_cache(self, record_id: Optional[str] = None):
        """Invalidate cache entries for this repository."""
        if not self.cache_manager:
            return

        if record_id:
            # Invalidate specific record cache
            cache_key = self._get_cache_key("get_by_id", record_id)
            await self.cache_manager.invalidate(cache_key)
        else:
            # Invalidate all cache entries for this table
            pattern = f"{self.table_name}:*"
            await self.cache_manager.invalidate_pattern(pattern)
